"""
TorchRL PPO training for Super Tic-Tac-Toe.

Demonstrates: EnvBase, ProbabilisticActor, ValueOperator, MaskedCategorical,
GAE, ClipPPOLoss, TensorDictReplayBuffer.

Uses a fast custom rollout for data collection (avoiding SyncDataCollector's
per-step overhead) while keeping TorchRL for all training computation.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.model import ActorCritic
from super_tictactoe.torchrl_env import SuperTicTacToeTorchEnv
from super_tictactoe.heuristics import (
    random_heuristic, greedy_agent, blocking_agent, horizontal_agent,
)


# ── Network module helpers ────────────────────────────────────────────────────

class _ActorHead(nn.Module):
    """Backbone + actor head. Passes action_mask through for MaskedCategorical."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self._ac = ac

    def forward(self, observation: torch.Tensor, action_mask: torch.Tensor):
        single = observation.dim() == 3
        if single:
            observation = observation.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        features = self._ac._backbone(observation)
        logits = self._ac.actor(features)
        if single:
            logits = logits.squeeze(0)
            action_mask = action_mask.squeeze(0)
        return logits, action_mask


class _CriticHead(nn.Module):
    """Backbone + critic head."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self._ac = ac

    def forward(self, observation: torch.Tensor, action_mask: torch.Tensor):
        single = observation.dim() == 3
        if single:
            observation = observation.unsqueeze(0)
        features = self._ac._backbone(observation)
        value = self._ac.critic(features)
        if single:
            value = value.squeeze(0)
        return value


def make_modules(ac: ActorCritic, action_spec):
    """
    Build a TorchRL ProbabilisticActor and ValueOperator from an ActorCritic.

    Both modules share the backbone inside `ac` — no weight duplication.

    Returns:
        actor: ProbabilisticActor  — writes 'action', 'sample_log_prob'
        critic: ValueOperator      — writes 'state_value'
    """
    actor_net = TensorDictModule(
        _ActorHead(ac),
        in_keys=["observation", "action_mask"],
        out_keys=["logits", "mask"],
    )
    actor = ProbabilisticActor(
        module=actor_net,
        in_keys=["logits", "mask"],
        out_keys=["action"],
        spec=action_spec,
        distribution_class=MaskedCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
        safe=False,
    )
    critic = ValueOperator(
        module=_CriticHead(ac),
        in_keys=["observation", "action_mask"],
    )
    return actor, critic


# ── Curriculum helpers ────────────────────────────────────────────────────────

def _fast_random_agent(env):
    """Pure random: pick any valid cell. Very fast (no board scanning)."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    return int(valid[np.random.randint(len(valid))])


CURRICULUM_OPPONENTS = [_fast_random_agent, horizontal_agent, greedy_agent, blocking_agent]
CURRICULUM_NAMES = ["random", "horizontal", "greedy", "blocking"]


# ── Fast vectorized rollout ───────────────────────────────────────────────────

def collect_rollout(
    ac: ActorCritic,
    opponent_fn,
    n_episodes: int = 256,
    success_rate: float = 0.5,
    selfplay_prob: float = 0.0,
):
    """
    Fast vectorized data collection: runs n_episodes games in parallel with
    batched forward passes. Agent (P1) vs opponent (P2).

    If selfplay_prob > 0, each game has that probability of being self-play
    (agent plays both sides) instead of using opponent_fn.

    Returns: (TensorDict, stats_dict)
        TensorDict: flat transitions ready for GAE + ClipPPOLoss
        stats_dict: {'wins': int, 'losses': int, 'draws': int}
    """
    envs = [SuperTicTacToeEnv(success_rate=success_rate) for _ in range(n_episodes)]
    states = [env.reset() for env in envs]
    done_flags = [False] * n_episodes
    # Decide per-game whether it's self-play
    is_selfplay = [np.random.random() < selfplay_prob for _ in range(n_episodes)]

    observations = []
    action_masks = []
    actions_list = []
    log_probs_list = []
    next_observations = []
    next_action_masks = []
    rewards = []
    dones = []

    wins = losses = draws = 0

    while not all(done_flags):
        active = [i for i in range(n_episodes) if not done_flags[i]]
        if not active:
            break

        # Batched forward pass for all active envs
        batch_obs = torch.FloatTensor(np.array([states[i] for i in active]))
        batch_masks = torch.BoolTensor(
            np.array([envs[i].get_action_mask() for i in active])
        )

        with torch.no_grad():
            probs, _ = ac(batch_obs, batch_masks)
            dist = torch.distributions.Categorical(probs)
            action_tensors = dist.sample()
            log_prob_tensors = dist.log_prob(action_tensors)

        # Execute actions and opponent moves
        for j, i in enumerate(active):
            obs = states[i].copy()
            mask = envs[i].get_action_mask().copy()
            action = action_tensors[j].item()
            log_prob = log_prob_tensors[j].item()

            # Agent plays
            next_state, reward, done, _ = envs[i].step(action)

            # Opponent plays if game not over
            if not done:
                if is_selfplay[i]:
                    # Self-play: use the model for P2 as well
                    opp_state = torch.FloatTensor(next_state).unsqueeze(0)
                    opp_mask = torch.BoolTensor(
                        envs[i].get_action_mask()
                    ).unsqueeze(0)
                    with torch.no_grad():
                        opp_probs, _ = ac(opp_state, opp_mask)
                        opp_dist = torch.distributions.Categorical(opp_probs)
                        opp_action = opp_dist.sample().item()
                else:
                    opp_action = opponent_fn(envs[i])
                next_state, _, done, _ = envs[i].step(opp_action)
                if done and envs[i].winner == 2:
                    reward = -1.0

            observations.append(obs)
            action_masks.append(mask)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            next_observations.append(next_state.copy())
            next_action_masks.append(envs[i].get_action_mask().copy())
            rewards.append(float(reward))
            dones.append(done)

            if done:
                done_flags[i] = True
                if envs[i].winner == 1:
                    wins += 1
                elif envs[i].winner == 2:
                    losses += 1
                else:
                    draws += 1
            else:
                states[i] = next_state

    n = len(observations)
    data = TensorDict(
        {
            "observation": torch.FloatTensor(np.array(observations)),
            "action_mask": torch.BoolTensor(np.array(action_masks)),
            "action": torch.LongTensor(actions_list),
            "sample_log_prob": torch.FloatTensor(log_probs_list),
            "next": TensorDict(
                {
                    "observation": torch.FloatTensor(np.array(next_observations)),
                    "action_mask": torch.BoolTensor(np.array(next_action_masks)),
                    "reward": torch.FloatTensor(rewards).unsqueeze(-1),
                    "done": torch.BoolTensor(dones).unsqueeze(-1),
                    "terminated": torch.BoolTensor(dones).unsqueeze(-1),
                },
                batch_size=[n],
            ),
        },
        batch_size=[n],
    )
    stats = {"wins": wins, "losses": losses, "draws": draws}
    return data, stats


# ── Evaluation helper ─────────────────────────────────────────────────────────

def eval_vs_opponent(ac: ActorCritic, opponent_fn, n_games: int = 100) -> float:
    env = SuperTicTacToeEnv(success_rate=0.5)
    wins = 0
    device = next(ac.parameters()).device
    for _ in range(n_games):
        state = env.reset()
        while not env.done:
            if env.current_player == 1:
                s = torch.FloatTensor(state).to(device)
                m = torch.BoolTensor(env.get_action_mask()).to(device)
                with torch.no_grad():
                    action, _, _ = ac.get_action(s, m)
            else:
                action = opponent_fn(env)
            state, _, _, _ = env.step(action)
        if env.winner == 1:
            wins += 1
    return wins / n_games


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    num_updates: int = 500,
    phase: int = 0,
    episodes_per_update: int = 128,
    ppo_epochs: int = 4,
    mini_batch_size: int = 512,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.05,
    critic_coef: float = 0.5,
    eval_every: int = 50,
    eval_games: int = 50,
    checkpoint_dir: str = "torchrl_ppo/checkpoints",
    resume: str = None,
    device: str = "cpu",
    selfplay_prob: float = 0.0,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device)

    phase_dir = os.path.join(checkpoint_dir, f"phase{phase}_{CURRICULUM_NAMES[phase]}")
    os.makedirs(phase_dir, exist_ok=True)

    env = SuperTicTacToeTorchEnv(opponent_fn=random_heuristic)

    ac = ActorCritic().to(device)
    actor, critic = make_modules(ac, env.action_spec)
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr)

    gae = GAE(value_network=critic, gamma=gamma, lmbda=lam)
    gae.set_keys(value="state_value", advantage="advantage", value_target="value_target")

    loss_fn = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_eps,
        entropy_bonus=True,
        entropy_coeff=entropy_coef,
        critic_coeff=critic_coef,
        normalize_advantage=True,
        loss_critic_type="l2",
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=50_000),
        batch_size=mini_batch_size,
    )

    current_phase = phase
    best_win_rate = 0.0
    start_update = 1

    history_updates = []
    history_win_rate = []
    history_actor_loss = []
    history_critic_loss = []
    history_eval_updates = []
    history_eval_win_rate = []

    if resume:
        ckpt = torch.load(resume, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ac.load_state_dict(ckpt["model"])
            saved_phase = ckpt.get("phase", phase)
            if phase != saved_phase:
                current_phase = phase
                best_win_rate = 0.0
                print(f"New phase detected — optimizer reset, history cleared.")
            else:
                optimizer.load_state_dict(ckpt["optimizer"])
                start_update = ckpt["update"] + 1
                current_phase = saved_phase
                best_win_rate = ckpt.get("best_win_rate", 0.0)
                history_updates = ckpt.get("history_updates", [])
                history_win_rate = ckpt.get("history_win_rate", [])
                history_actor_loss = ckpt.get("history_actor_loss", [])
                history_critic_loss = ckpt.get("history_critic_loss", [])
                history_eval_updates = ckpt.get("history_eval_updates", [])
                history_eval_win_rate = ckpt.get("history_eval_win_rate", [])
            print(f"Resumed from update {start_update - 1}, phase={CURRICULUM_NAMES[current_phase]}: {resume}")
        else:
            ac.load_state_dict(ckpt)
            print(f"Resumed weights only from: {resume}")

    opponent_fn = CURRICULUM_OPPONENTS[current_phase]
    n_phases = len(CURRICULUM_NAMES)
    print(f"Training phase {current_phase + 1}/{n_phases}: opponent = {CURRICULUM_NAMES[current_phase]}"
          + (f" + {selfplay_prob:.0%} self-play" if selfplay_prob > 0 else ""))

    for update in range(start_update, num_updates + 1):
        ac.cpu()
        data, stats = collect_rollout(
            ac, opponent_fn, n_episodes=episodes_per_update,
            selfplay_prob=selfplay_prob,
        )
        ac.to(device)
        data = data.to(device)

        with torch.no_grad():
            data = gae(data)

        replay_buffer.empty()
        replay_buffer.extend(data)

        actor_losses, critic_losses = [], []
        for _ in range(ppo_epochs):
            batch = replay_buffer.sample().to(device)
            loss_td = loss_fn(batch)
            loss = loss_td["loss_objective"] + loss_td["loss_entropy"] + loss_td["loss_critic"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            optimizer.step()
            actor_losses.append(loss_td["loss_objective"].item())
            critic_losses.append(loss_td["loss_critic"].item())

        n_eps = stats["wins"] + stats["losses"] + stats["draws"]
        history_updates.append(update)
        history_win_rate.append(stats["wins"] / n_eps)
        history_actor_loss.append(np.mean(actor_losses))
        history_critic_loss.append(np.mean(critic_losses))

        print(
            f"Update {update:4d}/{num_updates} | "
            f"phase={current_phase + 1}({CURRICULUM_NAMES[current_phase][:4]}) | "
            f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']} "
            f"(win={stats['wins']/n_eps:.0%}) | "
            f"actor={np.mean(actor_losses):.4f} | "
            f"critic={np.mean(critic_losses):.4f}",
            end="",
        )

        if update % 100 == 0:
            torch.save({
                "model": ac.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "phase": current_phase,
                "best_win_rate": best_win_rate,
                "history_updates": history_updates,
                "history_win_rate": history_win_rate,
                "history_actor_loss": history_actor_loss,
                "history_critic_loss": history_critic_loss,
                "history_eval_updates": history_eval_updates,
                "history_eval_win_rate": history_eval_win_rate,
            }, os.path.join(phase_dir, "checkpoint.pt"))

        if update % eval_every == 0:
            win_rate = eval_vs_opponent(ac, opponent_fn, n_games=eval_games)
            history_eval_updates.append(update)
            history_eval_win_rate.append(win_rate)
            print(f" | win_vs_{CURRICULUM_NAMES[current_phase]}={win_rate:.0%}", end="")
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(ac.state_dict(), os.path.join(phase_dir, "model_best.pt"))
                print(" <- best!", end="")

        print()

    final_ckpt = {
        "model": ac.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": num_updates,
        "phase": current_phase,
        "best_win_rate": best_win_rate,
        "history_updates": history_updates,
        "history_win_rate": history_win_rate,
        "history_actor_loss": history_actor_loss,
        "history_critic_loss": history_critic_loss,
        "history_eval_updates": history_eval_updates,
        "history_eval_win_rate": history_eval_win_rate,
    }
    torch.save(final_ckpt, os.path.join(phase_dir, "model_final.pt"))
    print(f"\nDone. Best win rate vs {CURRICULUM_NAMES[current_phase]}: {best_win_rate:.0%}")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(history_updates, history_win_rate, label=f"Win rate (vs {CURRICULUM_NAMES[current_phase]})", color="tab:blue")
    if history_eval_win_rate:
        axes[0].plot(history_eval_updates, history_eval_win_rate, "o-", label="Eval win rate", color="tab:orange")
    axes[0].set_ylabel("Win Rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title(f"TorchRL PPO — Phase {current_phase + 1} ({CURRICULUM_NAMES[current_phase]})")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history_updates, history_actor_loss, color="tab:red")
    axes[1].set_ylabel("Actor Loss")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(history_updates, history_critic_loss, color="tab:green")
    axes[2].set_ylabel("Critic Loss")
    axes[2].set_xlabel("Update")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(phase_dir, "training_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Training curve saved to: {plot_path}")

    return ac


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchRL PPO for Super Tic-Tac-Toe")
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Curriculum phase: 0=random, 1=greedy, 2=blocking, 3=horizontal")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="torchrl_ppo/checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--selfplay-prob", type=float, default=0.0,
                        help="Probability of self-play per game (0=pure heuristic, 1=pure self-play)")
    args = parser.parse_args()

    train(
        num_updates=args.num_updates,
        phase=args.phase,
        episodes_per_update=args.episodes,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch,
        lr=args.lr,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        device=args.device,
        selfplay_prob=args.selfplay_prob,
    )
