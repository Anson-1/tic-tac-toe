"""
TorchRL DQN training for Super Tic-Tac-Toe.

Value-based approach: learns Q(s,a) directly with epsilon-greedy exploration,
experience replay, and a target network for stability.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.heuristics import random_heuristic, greedy_agent, blocking_agent


# ── Q-Network ────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """CNN-based Q-network. Outputs Q-values for all 144 actions."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 12 * 12, 256)
        self.q_head = nn.Linear(256, 144)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, 12, 12) -> Q-values: (batch, 144)"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.q_head(x)

    def get_action(self, state: torch.Tensor, mask: torch.Tensor, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection with masking."""
        if np.random.random() < epsilon:
            valid = torch.where(mask)[0]
            return int(valid[np.random.randint(len(valid))].item())
        with torch.no_grad():
            q = self.forward(state.unsqueeze(0)).squeeze(0)
            q[~mask] = float('-inf')
            return int(q.argmax().item())


# ── Curriculum ────────────────────────────────────────────────────────────────

def _fast_random_agent(env):
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


# ── Data collection ───────────────────────────────────────────────────────────

def collect_episodes(
    q_net: QNetwork,
    opponent_fn,
    n_episodes: int = 64,
    epsilon: float = 0.1,
    success_rate: float = 1.0,
):
    """Collect transitions from n_episodes games. Returns TensorDict + stats."""
    envs = [SuperTicTacToeEnv(success_rate=success_rate) for _ in range(n_episodes)]
    states = [env.reset() for env in envs]
    done_flags = [False] * n_episodes

    observations, action_masks, actions = [], [], []
    next_observations, next_action_masks = [], []
    rewards, dones = [], []
    wins = losses = draws = 0

    while not all(done_flags):
        active = [i for i in range(n_episodes) if not done_flags[i]]
        if not active:
            break

        batch_obs = torch.FloatTensor(np.array([states[i] for i in active]))
        batch_masks = torch.BoolTensor(
            np.array([envs[i].get_action_mask() for i in active])
        )

        # Batched Q-values for epsilon-greedy
        with torch.no_grad():
            q_values = q_net(batch_obs)
            q_values[~batch_masks] = float('-inf')

        for j, i in enumerate(active):
            obs = states[i].copy()
            mask = envs[i].get_action_mask().copy()

            # Epsilon-greedy
            if np.random.random() < epsilon:
                valid = np.where(mask)[0]
                action = int(valid[np.random.randint(len(valid))])
            else:
                action = int(q_values[j].argmax().item())

            # Agent plays
            next_state, reward, done, _ = envs[i].step(action)

            # Opponent plays
            if not done:
                opp_action = opponent_fn(envs[i])
                next_state, _, done, _ = envs[i].step(opp_action)
                if done and envs[i].winner == 2:
                    reward = -1.0

            observations.append(obs)
            action_masks.append(mask)
            actions.append(action)
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
            "action": torch.LongTensor(actions),
            "next_observation": torch.FloatTensor(np.array(next_observations)),
            "next_action_mask": torch.BoolTensor(np.array(next_action_masks)),
            "reward": torch.FloatTensor(rewards),
            "done": torch.BoolTensor(dones),
        },
        batch_size=[n],
    )
    stats = {"wins": wins, "losses": losses, "draws": draws}
    return data, stats


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_vs_blocking(q_net: QNetwork, n_games: int = 100) -> float:
    env = SuperTicTacToeEnv(success_rate=0.5)
    wins = 0
    for _ in range(n_games):
        state = env.reset()
        while not env.done:
            if env.current_player == 1:
                s = torch.FloatTensor(state)
                m = torch.BoolTensor(env.get_action_mask())
                action = q_net.get_action(s, m, epsilon=0.0)
            else:
                action = blocking_agent(env)
            state, _, _, _ = env.step(action)
        if env.winner == 1:
            wins += 1
    return wins / n_games


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    num_updates: int = 1000,
    episodes_per_update: int = 64,
    batch_size: int = 256,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_frac: float = 0.5,
    target_update_freq: int = 10,
    buffer_size: int = 100_000,
    eval_every: int = 50,
    eval_games: int = 50,
    checkpoint_dir: str = "torchrl_dqn/checkpoints",
    resume: str = None,
    device: str = "cpu",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device)

    q_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    if resume:
        q_net.load_state_dict(torch.load(resume, map_location=device))
        print(f"Resumed from: {resume}")
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=buffer_size),
        batch_size=batch_size,
    )

    best_win_rate = 0.0
    current_phase = 0

    # Tracking for plots
    history_updates = []
    history_win_rate = []
    history_loss = []
    history_eval_updates = []
    history_eval_win_rate = []

    for update in range(1, num_updates + 1):
        # Curriculum
        new_phase = get_phase(update, num_updates)
        if new_phase != current_phase:
            current_phase = new_phase
            print(
                f"\n[Curriculum] Phase {current_phase + 1}: "
                f"opponent = {CURRICULUM_NAMES[current_phase]}"
            )

        opponent_fn = CURRICULUM_OPPONENTS[current_phase]

        # Epsilon schedule
        decay_updates = int(num_updates * epsilon_decay_frac)
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * update / decay_updates
        )

        # Collect data (on CPU since env is CPU-based)
        q_net_cpu = q_net.cpu()
        data, stats = collect_episodes(
            q_net_cpu, opponent_fn, n_episodes=episodes_per_update, epsilon=epsilon
        )
        q_net.to(device)
        replay_buffer.extend(data)

        # Train on mini-batches
        losses = []
        if len(replay_buffer) >= batch_size:
            for _ in range(4):
                batch = replay_buffer.sample().to(device)
                obs = batch["observation"]
                actions = batch["action"]
                rewards_b = batch["reward"]
                next_obs = batch["next_observation"]
                next_mask = batch["next_action_mask"]
                done_b = batch["done"].float()

                # Current Q-values
                q_values = q_net(obs)
                q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Target Q-values (Double DQN style)
                with torch.no_grad():
                    next_q = target_net(next_obs)
                    next_q[~next_mask] = float('-inf')
                    next_q_max = next_q.max(dim=1).values
                    target = rewards_b + gamma * next_q_max * (1 - done_b)

                loss = F.mse_loss(q_selected, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

        # Update target network
        if update % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        n_eps = stats["wins"] + stats["losses"] + stats["draws"]
        avg_loss = np.mean(losses) if losses else 0.0

        # Track metrics
        history_updates.append(update)
        history_win_rate.append(stats["wins"] / n_eps)
        history_loss.append(avg_loss)

        print(
            f"Update {update:4d}/{num_updates} | "
            f"phase={current_phase + 1}({CURRICULUM_NAMES[current_phase][:4]}) | "
            f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']} "
            f"(win={stats['wins']/n_eps:.0%}) | "
            f"eps={epsilon:.2f} | "
            f"loss={avg_loss:.4f}",
            end="",
        )

        # Evaluation
        if update % eval_every == 0:
            win_rate = eval_vs_blocking(q_net, n_games=eval_games)
            history_eval_updates.append(update)
            history_eval_win_rate.append(win_rate)
            print(f" | win_vs_blocking={win_rate:.0%}", end="")
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(
                    q_net.state_dict(),
                    os.path.join(checkpoint_dir, "model_best.pt"),
                )
                print(" <- best!", end="")

        print()

    # Save final checkpoint
    torch.save(q_net.state_dict(), os.path.join(checkpoint_dir, "model_final.pt"))
    print(f"\nDone. Best win rate vs blocking: {best_win_rate:.0%}")

    # Save training curve
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(history_updates, history_win_rate, label="Win rate (vs curriculum opp)", color="tab:blue")
    if history_eval_win_rate:
        axes[0].plot(history_eval_updates, history_eval_win_rate, "o-", label="Win rate (vs blocking)", color="tab:orange")
    axes[0].set_ylabel("Win Rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title("TorchRL DQN Training — Super Tic-Tac-Toe")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_updates, history_loss, color="tab:red")
    axes[1].set_ylabel("Q-Loss (MSE)")
    axes[1].set_xlabel("Update")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "training_curve.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Training curve saved to: {plot_path}")

    return q_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchRL DQN for Super Tic-Tac-Toe")
    parser.add_argument("--num-updates", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="torchrl_dqn/checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train(
        num_updates=args.num_updates,
        episodes_per_update=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        device=args.device,
    )
