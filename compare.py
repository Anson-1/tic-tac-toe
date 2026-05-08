"""
Evaluation utilities: play games between any two agents and report results.
Used by alphazero/eval_az.py, ppo_curriculum/eval_ppo_progress.py, etc.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.model import ActorCritic
from super_tictactoe.mcts import MCTS
from torchrl_dqn.torchrl_dqn_train import QNetwork
from super_tictactoe.heuristics import (
    greedy_agent,
    blocking_agent,
    safe_agent,
    horizontal_agent,
)


def random_agent(env):
    """Pick a random valid cell."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    return int(valid[np.random.randint(len(valid))])


def model_agent(model, device='cpu'):
    """Wrap an ActorCritic model into an agent callable (env) -> action."""
    def agent(env):
        state = torch.FloatTensor(env._get_state()).to(device)
        mask = torch.BoolTensor(env.get_action_mask()).to(device)
        with torch.no_grad():
            action, _, _ = model.get_action(state, mask)
        return action
    return agent


def dqn_agent(model, device='cpu'):
    """Wrap a QNetwork model into an agent callable (env) -> action."""
    def agent(env):
        state = torch.FloatTensor(env._get_state()).to(device)
        mask = torch.BoolTensor(env.get_action_mask()).to(device)
        return model.get_action(state, mask, epsilon=0.0)
    return agent


def mcts_agent_factory(model, device='cpu', num_simulations=25):
    """Wrap an ActorCritic model + MCTS into an agent callable (env) -> action."""
    mcts = MCTS(model, device=device, num_simulations=num_simulations)
    def agent(env):
        return mcts.get_action(env)
    return agent


def play_games(agent1_fn, agent2_fn, n_games, success_rate=0.5):
    """
    Play n_games between agent1 (P1) and agent2 (P2).

    Returns: (p1_win_rate, p2_win_rate, draw_rate, avg_steps)
    """
    p1_wins = p2_wins = draws = 0
    total_steps = 0

    for _ in range(n_games):
        env = SuperTicTacToeEnv(success_rate=success_rate)
        env.reset()
        steps = 0

        while not env.done:
            if env.current_player == 1:
                action = agent1_fn(env)
            else:
                action = agent2_fn(env)
            env.step(action)
            steps += 1

        total_steps += steps
        if env.winner == 1:
            p1_wins += 1
        elif env.winner == 2:
            p2_wins += 1
        else:
            draws += 1

    n = max(n_games, 1)
    return p1_wins / n, p2_wins / n, draws / n, total_steps / n


def load_model(path, device='cpu', model_type='actor_critic'):
    """Load a model checkpoint."""
    if model_type == 'dqn':
        m = QNetwork().to(device)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            m.load_state_dict(state['model'])
        else:
            m.load_state_dict(state)
    else:
        m = ActorCritic().to(device)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            m.load_state_dict(state['model'])
        else:
            m.load_state_dict(state)
    m.eval()
    return m


if __name__ == '__main__':
    import os
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Tournament: models vs heuristics")
    parser.add_argument('--games', type=int, default=200,
                        help='Games per matchup (split evenly as P1/P2)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default='tournament_results.png',
                        help='Output plot path')
    args = parser.parse_args()

    N = args.games // 2  # play half as P1, half as P2

    # Discover available model checkpoints: (path, model_type)
    model_entries = {
        'PPO-baseline': ('ppo_baseline/checkpoints/model_final.pt', 'actor_critic'),
        'PPO-curriculum': ('ppo_curriculum/checkpoints/model_final.pt', 'actor_critic'),
        'PPO-curr+MCTS': ('ppo_curriculum/checkpoints/model_final.pt', 'mcts'),
        'AlphaZero': ('alphazero/checkpoints/model_best.pt', 'actor_critic'),
        'TorchRL-PPO': ('torchrl_ppo/checkpoints/phase0_random/model_best.pt', 'actor_critic'),
        'TorchRL-DQN': ('torchrl_dqn/checkpoints/phase0_random/model_best.pt', 'dqn'),
    }

    heuristic_list = [
        ('Random', random_agent),
        ('Greedy', greedy_agent),
        ('Blocking', blocking_agent),
        ('Safe', safe_agent),
        ('Horizontal', horizontal_agent),
    ]

    print(f"{'Model':<18} {'Opponent':<12} {'Win':>6} {'Loss':>6} {'Draw':>6} {'Steps':>6}")
    print("-" * 60)

    # Collect results for plotting: {model_name: [win_rate_vs_each_heuristic]}
    results = {}

    for model_name, (path, mtype) in model_entries.items():
        if not os.path.exists(path):
            print(f"{model_name:<18} (checkpoint not found: {path})")
            continue

        model = load_model(path, device=args.device,
                           model_type='actor_critic' if mtype == 'mcts' else mtype)
        if mtype == 'dqn':
            agent = dqn_agent(model, device=args.device)
        elif mtype == 'mcts':
            agent = mcts_agent_factory(model, device=args.device, num_simulations=25)
        else:
            agent = model_agent(model, device=args.device)
        results[model_name] = []

        for h_name, h_fn in heuristic_list:
            # Agent as P1
            w1, l1, d1, s1 = play_games(agent, h_fn, N)
            # Agent as P2
            l2, w2, d2, s2 = play_games(h_fn, agent, N)
            # Average both sides
            win = (w1 + w2) / 2
            loss = (l1 + l2) / 2
            draw = (d1 + d2) / 2
            steps = (s1 + s2) / 2
            results[model_name].append(win)
            print(f"{model_name:<18} {h_name:<12} {win:>5.0%} {loss:>5.0%} {draw:>5.0%} {steps:>6.1f}")
        print()

    # ── Plot: grouped bar chart ──────────────────────────────────────────────
    if not results:
        print("No models found, skipping plot.")
    else:
        h_names = [name for name, _ in heuristic_list]
        model_names = list(results.keys())
        n_models = len(model_names)
        n_heuristics = len(h_names)

        x = np.arange(n_heuristics)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']

        for i, model_name in enumerate(model_names):
            win_rates = [r * 100 for r in results[model_name]]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, win_rates, width, label=model_name,
                          color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)
            # Add value labels on bars
            for bar in bars:
                h = bar.get_height()
                if h > 5:
                    ax.text(bar.get_x() + bar.get_width() / 2, h - 4, f'{h:.0f}%',
                            ha='center', va='top', fontsize=8, color='white', fontweight='bold')

        ax.set_xlabel('Opponent Heuristic', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Tournament Results: RL Agents vs Heuristic Opponents', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(h_names, fontsize=11)
        ax.set_ylim(0, 105)
        ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        fig.savefig(args.out, dpi=150)
        plt.close(fig)
        print(f"Plot saved to: {args.out}")
