"""
Comprehensive comparison of trained agents.
Usage: conda run -n tictactoe python compare.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from super_tictactoe.model import ActorCritic
from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.mcts import MCTS

N_GAMES = 200  # games per matchup


# ── Helpers ──────────────────────────────────────────────────────────────────

def load(path, device='cpu'):
    m = ActorCritic().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def play_games(agent1, agent2, n=N_GAMES):
    """
    Play n games: agent1 as P1, agent2 as P2.
    Each agent is callable: action = agent(env).
    Returns (p1_wins, p2_wins, draws, avg_steps).
    """
    p1_wins = p2_wins = draws = 0
    total_steps = 0

    for _ in range(n):
        env = SuperTicTacToeEnv()
        env.reset()
        steps = 0

        while not env.done:
            agent = agent1 if env.current_player == 1 else agent2
            action = agent(env)
            env.step(action)
            steps += 1

        total_steps += steps
        if env.winner == 1:   p1_wins += 1
        elif env.winner == 2: p2_wins += 1
        else:                 draws += 1

    return p1_wins / n, p2_wins / n, draws / n, total_steps / n


def model_agent(model, device='cpu'):
    """Wrap a model as a callable agent."""
    def agent(env):
        state = torch.FloatTensor(env._get_state()).unsqueeze(0).to(device)
        mask  = torch.BoolTensor(env.get_action_mask()).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _ = model.get_action(state.squeeze(0), mask.squeeze(0))
        return action
    return agent


def mcts_agent(model, simulations=50, device='cpu'):
    """Wrap a model+MCTS as a callable agent."""
    mcts = MCTS(model, device=device, num_simulations=simulations)
    def agent(env):
        return mcts.get_action(env)
    return agent


def random_agent(env):
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    return int(np.random.choice(valid))


# ── Elo ───────────────────────────────────────────────────────────────────────

def compute_elo(results, k=32, initial=1000):
    """
    results: list of (name_a, name_b, score_a)  where score_a in [0, 0.5, 1]
    Returns dict of name -> elo rating.
    """
    names = list({n for r in results for n in r[:2]})
    elo = {n: initial for n in names}

    for name_a, name_b, score_a in results:
        ea = 1 / (1 + 10 ** ((elo[name_b] - elo[name_a]) / 400))
        eb = 1 - ea
        elo[name_a] += k * (score_a - ea)
        elo[name_b] += k * ((1 - score_a) - eb)

    return elo


# ── Main comparison ───────────────────────────────────────────────────────────

def main():
    # Discover available models
    agents_cfg = []

    if os.path.exists('checkpoints/model_final.pt'):
        agents_cfg.append(('PPO-baseline',    load('checkpoints/model_final.pt'),        False))
    if os.path.exists('checkpoints_ppo_cl/model_final.pt'):
        agents_cfg.append(('PPO-curriculum',  load('checkpoints_ppo_cl/model_final.pt'), False))
    if os.path.exists('checkpoints_az/model_best.pt'):
        agents_cfg.append(('AZ-best',         load('checkpoints_az/model_best.pt'),      False))
    if os.path.exists('checkpoints_az/model_final.pt'):
        agents_cfg.append(('AZ-final',        load('checkpoints_az/model_final.pt'),     False))
    if os.path.exists('checkpoints_az_cl/model_best.pt'):
        agents_cfg.append(('AZ-curriculum',   load('checkpoints_az_cl/model_best.pt'),   False))

    if not agents_cfg:
        print("No checkpoints found.")
        return

    print(f"Agents: {[a[0] for a in agents_cfg]}")
    print(f"Games per matchup: {N_GAMES}\n")

    # Build callable agents
    def make_agent(cfg):
        name, model, use_mcts = cfg
        if name == 'Random':
            return random_agent
        if use_mcts:
            return mcts_agent(model, simulations=20)
        return model_agent(model)

    # ── 1. Win rate vs random ─────────────────────────────────────────────────
    print("=" * 60)
    print("1. WIN RATE vs RANDOM BASELINE")
    print("=" * 60)
    vs_random = {}
    for cfg in agents_cfg:
        name = cfg[0]
        agent = make_agent(cfg)
        print(f"  {name:20s} evaluating...", end='\r', flush=True)
        p1, p2, d, steps = play_games(agent, random_agent, N_GAMES)
        vs_random[name] = {'win': p1, 'loss': p2, 'draw': d, 'steps': steps}
        print(f"  {name:20s} | win={p1:.0%}  loss={p2:.0%}  draw={d:.0%}  avg_steps={steps:.1f}")

    # ── 2. Head-to-head round-robin ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. HEAD-TO-HEAD ROUND-ROBIN")
    print("=" * 60)
    elo_results = []
    h2h = {}

    for i, cfg_a in enumerate(agents_cfg):
        for j, cfg_b in enumerate(agents_cfg):
            if i >= j:
                continue
            name_a, name_b = cfg_a[0], cfg_b[0]
            agent_a = make_agent(cfg_a)
            agent_b = make_agent(cfg_b)

            # Play both directions to eliminate first-mover bias
            print(f"  {name_a} vs {name_b} (fwd)...", end='\r', flush=True)
            p1, p2, d, _ = play_games(agent_a, agent_b, N_GAMES // 2)
            print(f"  {name_a} vs {name_b} (rev)...", end='\r', flush=True)
            p2b, p1b, db, _ = play_games(agent_b, agent_a, N_GAMES // 2)

            a_wins = (p1 + p1b) / 2
            b_wins = (p2 + p2b) / 2
            draws  = (d  + db)  / 2
            h2h[(name_a, name_b)] = (a_wins, b_wins, draws)
            print(f"  {name_a:20s} vs {name_b:20s} | "
                  f"{name_a}={a_wins:.0%}  {name_b}={b_wins:.0%}  draw={draws:.0%}")

            for _ in range(int(a_wins * N_GAMES)):
                elo_results.append((name_a, name_b, 1.0))
            for _ in range(int(b_wins * N_GAMES)):
                elo_results.append((name_a, name_b, 0.0))
            for _ in range(int(draws * N_GAMES)):
                elo_results.append((name_a, name_b, 0.5))

    # Add random as baseline in Elo
    for cfg in agents_cfg:
        name = cfg[0]
        w = vs_random[name]['win']
        l = vs_random[name]['loss']
        d = vs_random[name]['draw']
        for _ in range(int(w * N_GAMES)):
            elo_results.append((name, 'Random', 1.0))
        for _ in range(int(l * N_GAMES)):
            elo_results.append((name, 'Random', 0.0))
        for _ in range(int(d * N_GAMES)):
            elo_results.append((name, 'Random', 0.5))

    # ── 3. Elo ratings ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. ELO RATINGS (higher = stronger)")
    print("=" * 60)
    elo = compute_elo(elo_results)
    for name, rating in sorted(elo.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} | Elo={rating:.0f}")

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Win rate vs random
    names = [c[0] for c in agents_cfg]
    wins  = [vs_random[n]['win'] * 100 for n in names]
    axes[0].bar(names, wins, color='steelblue')
    axes[0].axhline(50, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Win rate vs random (%)')
    axes[0].set_title('Win Rate vs Random Baseline')
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis='x', rotation=20)

    # Elo
    elo_names   = [n for n, _ in sorted(elo.items(), key=lambda x: -x[1])]
    elo_ratings = [r for _, r in sorted(elo.items(), key=lambda x: -x[1])]
    colors = ['gold' if n == 'Random' else 'steelblue' for n in elo_names]
    axes[1].barh(elo_names, elo_ratings, color=colors)
    axes[1].set_xlabel('Elo rating')
    axes[1].set_title('Elo Ratings (round-robin)')

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    print("\nSaved comparison.png")


if __name__ == '__main__':
    main()
