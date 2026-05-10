"""
Training process analysis:
  1. Smoothed training curves (PPO + DQN)
  2. Policy heatmap on empty board
  3. Value estimates across board positions
  4. Policy entropy over training
  5. Q-value distribution (DQN)
  6. Cross-model value comparison on key positions
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from super_tictactoe.env import SuperTicTacToeEnv, GRID_POSITIONS
from super_tictactoe.model import ActorCritic
from torchrl_dqn.torchrl_dqn_train import QNetwork


# ── helpers ───────────────────────────────────────────────────────────────────

def smooth(values, window=15):
    """Simple moving average."""
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]


def load_ppo(path):
    model = ActorCritic()
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_dqn(path):
    q_net = QNetwork()
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    q_net.load_state_dict(state_dict)
    q_net.eval()
    return q_net


def get_policy_value(model, env):
    s = torch.FloatTensor(env._get_state()).unsqueeze(0)
    m = torch.BoolTensor(env.get_action_mask()).unsqueeze(0)
    with torch.no_grad():
        probs, value = model(s, m)
    return probs[0].numpy(), value[0].item()


def entropy(probs):
    p = probs[probs > 1e-10]
    return -np.sum(p * np.log(p))


def valid_mask():
    env = SuperTicTacToeEnv()
    return env.valid_mask


# ── 1. Training curves ────────────────────────────────────────────────────────

def plot_training_curves():
    ppo_ckpt = torch.load(
        'torchrl_ppo/checkpoints/phase0_random/checkpoint.pt', map_location='cpu'
    )
    dqn_ckpt = torch.load(
        'torchrl_dqn/checkpoints/phase0_random/model_final.pt', map_location='cpu'
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Training Dynamics', fontsize=14, fontweight='bold')

    # PPO win rate
    ax = axes[0, 0]
    updates = ppo_ckpt['history_updates']
    wr = ppo_ckpt['history_win_rate']
    ax.plot(updates, wr, alpha=0.25, color='tab:blue', linewidth=0.8)
    ax.plot(updates, smooth(wr), color='tab:blue', linewidth=2, label='Smoothed')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='50% baseline')
    ax.set_title('TorchRL PPO — Win Rate vs Random')
    ax.set_ylabel('Win Rate')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PPO critic loss
    ax = axes[0, 1]
    cl = ppo_ckpt['history_critic_loss']
    ax.plot(updates, cl, alpha=0.25, color='tab:green', linewidth=0.8)
    ax.plot(updates, smooth(cl), color='tab:green', linewidth=2, label='Critic Loss (MSE)')
    ax.set_title('TorchRL PPO — Critic Loss (Value Convergence)')
    ax.set_ylabel('MSE Loss  E[(V(s) − V*target)²]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PPO policy entropy from actor loss
    ax = axes[1, 0]
    al = np.array(ppo_ckpt['history_actor_loss'])
    ax.plot(updates, al, alpha=0.25, color='tab:red', linewidth=0.8)
    ax.plot(updates, smooth(al), color='tab:red', linewidth=2, label='Actor Loss')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title('TorchRL PPO — Actor Loss\n(bounded near 0 by PPO clipping ε=0.2)')
    ax.set_ylabel('Clipped Surrogate Loss')
    ax.set_xlabel('Update')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # DQN Q-loss
    ax = axes[1, 1]
    d_updates = dqn_ckpt['history_updates']
    d_loss = dqn_ckpt['history_loss']
    ax.plot(d_updates, d_loss, alpha=0.25, color='tab:orange', linewidth=0.8)
    ax.plot(d_updates, smooth(d_loss), color='tab:orange', linewidth=2, label='Q-Loss (MSE)')
    ax.set_title('TorchRL DQN — Bellman Residual\nE[(r + γ max Q_target(s\',a\') − Q(s,a))²]')
    ax.set_ylabel('MSE Loss')
    ax.set_xlabel('Update')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('analysis_training_curves.png', dpi=150)
    plt.close(fig)
    print("Saved: analysis_training_curves.png")


# ── 2. Policy heatmap on empty board ─────────────────────────────────────────

def plot_policy_heatmaps():
    models = {
        'PPO Baseline': load_ppo('ppo_baseline/checkpoints/model_final.pt'),
        'PPO Curriculum': load_ppo('ppo_curriculum/checkpoints/model_final.pt'),
        'AlphaZero': load_ppo('alphazero/checkpoints/model_best.pt'),
    }

    env = SuperTicTacToeEnv()
    env.reset()
    mask = env.valid_mask

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Policy π(a|s) on Empty Board — Preferred Opening Moves', fontsize=13, fontweight='bold')

    for ax, (name, model) in zip(axes, models.items()):
        probs, value = get_policy_value(model, env)
        heatmap = probs.reshape(12, 12)
        heatmap[~mask] = np.nan

        im = ax.imshow(heatmap, cmap='hot', vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Draw sub-grid boundaries
        for r, c in GRID_POSITIONS:
            rect = plt.Rectangle((c - 0.5, r - 0.5), 4, 4,
                                  fill=False, edgecolor='cyan', linewidth=1.5)
            ax.add_patch(rect)

        ax.set_title(f'{name}\nV(s₀) = {value:.3f}  H(π) = {entropy(probs):.2f}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig('analysis_policy_heatmap.png', dpi=150)
    plt.close(fig)
    print("Saved: analysis_policy_heatmap.png")


# ── 3. DQN Q-value distribution ──────────────────────────────────────────────

def plot_qvalue_distribution():
    """
    Show Q-value distribution across all 144 actions for key board states.
    Compares masked (invalid=-inf) vs valid action Q-values.
    """
    q_net = load_dqn('torchrl_dqn/checkpoints/phase0_random/model_final.pt')

    # Three positions: empty board, mid-game, near-win
    positions = {}

    # Empty board
    env = SuperTicTacToeEnv(success_rate=1.0)
    env.reset()
    positions['Empty Board'] = (
        torch.FloatTensor(env._get_state()),
        torch.BoolTensor(env.get_action_mask())
    )

    # Mid-game: place a few pieces
    env2 = SuperTicTacToeEnv(success_rate=1.0)
    env2.reset()
    for a in [4*12+4, 8*12+0, 4*12+5, 8*12+1]:
        if not env2.done:
            env2.board[a//12, a%12] = env2.current_player
            env2.current_player = 3 - env2.current_player
    positions['Mid-game'] = (
        torch.FloatTensor(env2._get_state()),
        torch.BoolTensor(env2.get_action_mask())
    )

    # Near-win: P1 has 3-in-a-row
    env3 = SuperTicTacToeEnv(success_rate=1.0)
    env3.reset()
    for c in [4, 5, 6]:
        env3.board[0, c] = 1
    for c in [0, 1]:
        env3.board[8, c] = 2
    positions['P1 Near-Win\n(3-in-a-row row 0)'] = (
        torch.FloatTensor(env3._get_state()),
        torch.BoolTensor(env3.get_action_mask())
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('DQN Q-Value Distribution Q(s,a) Across All Actions', fontsize=13, fontweight='bold')

    for ax, (name, (state, mask)) in zip(axes, positions.items()):
        with torch.no_grad():
            q_vals = q_net(state.unsqueeze(0)).squeeze(0).numpy()

        valid_q = q_vals[mask.numpy()]
        invalid_q = q_vals[~mask.numpy()]

        ax.hist(valid_q, bins=20, color='tab:blue', alpha=0.7, label=f'Valid ({mask.sum()} actions)')
        ax.hist(invalid_q, bins=20, color='tab:red', alpha=0.4, label=f'Invalid ({(~mask).sum()} actions)')
        ax.axvline(valid_q.max(), color='blue', linestyle='--', linewidth=1.5,
                   label=f'max Q = {valid_q.max():.3f}')
        ax.axvline(valid_q.mean(), color='cyan', linestyle=':', linewidth=1.5,
                   label=f'mean Q = {valid_q.mean():.3f}')
        ax.set_title(name)
        ax.set_xlabel('Q(s, a)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig('analysis_qvalue_distribution.png', dpi=150)
    plt.close(fig)
    print("Saved: analysis_qvalue_distribution.png")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Running training process analysis...")
    plot_training_curves()
    plot_policy_heatmaps()
    plot_qvalue_distribution()
    print("\nAll analysis plots saved.")
