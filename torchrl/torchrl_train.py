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
    random_heuristic, greedy_agent, blocking_agent,
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


CURRICULUM_OPPONENTS = [_fast_random_agent, greedy_agent, blocking_agent]
CURRICULUM_NAMES = ["random", "greedy", "blocking"]


def get_phase(update: int, total_updates: int) -> int:
    if update < total_updates / 3:
        return 0
    if update < 2 * total_updates / 3:
        return 1
    return 2


# ── Fast vectorized rollout ───────────────────────────────────────────────────

def collect_rollout(
    ac: ActorCritic,
    opponent_fn,
    n_episodes: int = 256,
    success_rate: float = 0.5,
):
    """
    Fast vectorized data collection: runs n_episodes games in parallel with
    batched forward passes. Agent (P1) vs heuristic (P2).

    Returns: (TensorDict, stats_dict)
        TensorDict: flat transitions ready for GAE + ClipPPOLoss
        stats_dict: {'wins': int, 'losses': int, 'draws': int}
    """
    envs = [SuperTicTacToeEnv(success_rate=success_rate) for _ in range(n_episodes)]
    states = [env.reset() for env in envs]
    done_flags = [False] * n_episodes

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

def eval_vs_blocking(ac: ActorCritic, n_games: int = 100) -> float:
    """Win rate of ac (as P1) vs blocking_agent over n_games."""
    env = SuperTicTacToeEnv(success_rate=0.5)
    wins = 0
    for _ in range(n_games):
        state = env.reset()
        while not env.done:
            if env.current_player == 1:
                s = torch.FloatTensor(state)
                m = torch.BoolTensor(env.get_action_mask())
                with torch.no_grad():
                    action, _, _ = ac.get_action(s, m)
            else:
                action = blocking_agent(env)
            state, _, _, _ = env.step(action)
        if env.winner == 1:
            wins += 1
    return wins / n_games


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    num_updates: int = 200,
    episodes_per_update: int = 64,
    ppo_epochs: int = 4,
    mini_batch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.05,
    critic_coef: float = 0.5,
    eval_every: int = 50,
    eval_games: int = 50,
    checkpoint_dir: str = "checkpoints_torchrl",
    resume: str = None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Environment (for spec extraction) ────────────────────────────────────
    env = SuperTicTacToeTorchEnv(opponent_fn=random_heuristic)

    # ── Model ────────────────────────────────────────────────────────────────
    ac = ActorCritic()
    if resume:
        ac.load_state_dict(torch.load(resume, map_location="cpu"))
        print(f"Resumed from: {resume}")
    actor, critic = make_modules(ac, env.action_spec)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(ac.parameters(), lr=lr)

    # ── TorchRL components ───────────────────────────────────────────────────
    gae = GAE(
        value_network=critic,
        gamma=gamma,
        lmbda=lam,
    )
    gae.set_keys(
        value="state_value",
        advantage="advantage",
        value_target="value_target",
    )

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

    # ── Loop ─────────────────────────────────────────────────────────────────
    best_win_rate = 0.0
    current_phase = 0

    # Tracking for plots
    history_updates = []
    history_win_rate = []
    history_actor_loss = []
    history_critic_loss = []
    history_eval_updates = []
    history_eval_win_rate = []

    for update in range(1, num_updates + 1):
        # Curriculum: switch opponent when phase boundary is crossed
        new_phase = get_phase(update, num_updates)
        if new_phase != current_phase:
            current_phase = new_phase
            print(
                f"\n[Curriculum] Phase {current_phase + 1}: "
                f"opponent = {CURRICULUM_NAMES[current_phase]}"
            )

        opponent_fn = CURRICULUM_OPPONENTS[current_phase]

        # ── Collect data (fast custom rollout) ───────────────────────────────
        data, stats = collect_rollout(ac, opponent_fn, n_episodes=episodes_per_update)

        # ── Compute GAE advantages ───────────────────────────────────────────
        with torch.no_grad():
            data = gae(data)

        # ── Fill replay buffer ───────────────────────────────────────────────
        replay_buffer.empty()
        replay_buffer.extend(data)

        # ── PPO mini-batch updates ───────────────────────────────────────────
        actor_losses, critic_losses = [], []
        for _ in range(ppo_epochs):
            batch = replay_buffer.sample()
            loss_td = loss_fn(batch)
            loss = (
                loss_td["loss_objective"]
                + loss_td["loss_entropy"]
                + loss_td["loss_critic"]
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            optimizer.step()
            actor_losses.append(loss_td["loss_objective"].item())
            critic_losses.append(loss_td["loss_critic"].item())

        n_eps = stats["wins"] + stats["losses"] + stats["draws"]
        print(
            f"Update {update:4d}/{num_updates} | "
            f"steps={len(data)} | "
            f"phase={current_phase + 1}({CURRICULUM_NAMES[current_phase][:4]}) | "
            f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']} "
            f"(win={stats['wins']/n_eps:.0%}) | "
            f"actor={np.mean(actor_losses):.4f} | "
            f"critic={np.mean(critic_losses):.4f}",
            end="",
        )

        # Track metrics
        history_updates.append(update)
        history_win_rate.append(stats["wins"] / n_eps)
        history_actor_loss.append(np.mean(actor_losses))
        history_critic_loss.append(np.mean(critic_losses))

        # ── Periodic evaluation + best-checkpoint saving ─────────────────────
        if update % eval_every == 0:
            win_rate = eval_vs_blocking(ac, n_games=eval_games)
            history_eval_updates.append(update)
            history_eval_win_rate.append(win_rate)
            print(f" | win_vs_blocking={win_rate:.0%}", end="")
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(
                    ac.state_dict(),
                    os.path.join(checkpoint_dir, "model_best.pt"),
                )
                print(" <- best!", end="")

        print()

    # Save final checkpoint
    torch.save(ac.state_dict(), os.path.join(checkpoint_dir, "model_final.pt"))
    print(f"\nDone. Best win rate vs blocking: {best_win_rate:.0%}")
    print(f"Checkpoints in: {checkpoint_dir}/")

    # ── Save training curve plot ─────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(history_updates, history_win_rate, label="Win rate (vs curriculum opp)", color="tab:blue")
    if history_eval_win_rate:
        axes[0].plot(history_eval_updates, history_eval_win_rate, "o-", label="Win rate (vs blocking)", color="tab:orange")
    axes[0].set_ylabel("Win Rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title("TorchRL PPO Training — Super Tic-Tac-Toe")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_updates, history_actor_loss, color="tab:red")
    axes[1].set_ylabel("Actor Loss (PPO objective)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history_updates, history_critic_loss, color="tab:green")
    axes[2].set_ylabel("Critic Loss")
    axes[2].set_xlabel("Update")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "training_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Training curve saved to: {plot_path}")

    return ac


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TorchRL PPO for Super Tic-Tac-Toe"
    )
    parser.add_argument("--num-updates", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_torchrl")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. model_final.pt)")
    args = parser.parse_args()

    train(
        num_updates=args.num_updates,
        episodes_per_update=args.episodes,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch,
        lr=args.lr,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )
