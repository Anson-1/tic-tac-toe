import numpy as np
import torch
from typing import Dict, List

from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.model import ActorCritic
from super_tictactoe.ppo import compute_gae


def collect_episode(
    env: SuperTicTacToeEnv, model: ActorCritic, device: str = 'cpu'
) -> List[Dict]:
    """
    Run one self-play game. Returns list of step dicts for both players.
    Reward: +1 for winner's last action, -1 for loser's last action, 0 otherwise.
    """
    state = env.reset()
    episode = []

    while not env.done:
        player = env.current_player
        action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
        state_tensor = torch.FloatTensor(state).to(device)

        with torch.no_grad():
            action, log_prob, value = model.get_action(state_tensor, action_mask)

        next_state, reward, done, info = env.step(action)

        episode.append({
            'state': state,
            'action_mask': action_mask.cpu().numpy(),
            'action': action,
            'log_prob': log_prob.item(),
            'value': value.item(),
            'reward': reward,
            'done': done,
            'player': player,
        })
        state = next_state

    # Retroactively assign -1 to loser's last move
    if env.winner is not None:
        loser = 3 - env.winner
        for i in range(len(episode) - 1, -1, -1):
            if episode[i]['player'] == loser:
                episode[i]['reward'] = -1.0
                break

    return episode


def collect_episodes_vectorized(
    n_envs: int,
    model: ActorCritic,
    device: str = 'cpu',
    opponent_model: ActorCritic = None,
    success_rate: float = 0.5,
) -> List[List[Dict]]:
    """
    Run n_envs games simultaneously with one batched forward pass per step.
    Returns list of n_envs episodes, each a list of step dicts.

    If opponent_model is provided, P1 uses model and P2 uses opponent_model.
    Only P1 steps are stored in the training buffer (opponent is frozen).
    If opponent_model is None, model plays both sides (standard self-play).
    """
    THREAT_PENALTY_COEF = 0.3

    envs = [SuperTicTacToeEnv(success_rate=success_rate) for _ in range(n_envs)]
    states = [env.reset() for env in envs]
    episodes: List[List[Dict]] = [[] for _ in range(n_envs)]
    done_flags = [False] * n_envs
    # Accumulated threat-growth penalty to apply to P1's next recorded step
    pending_threat_penalty = [0.0] * n_envs

    while not all(done_flags):
        active = [i for i, d in enumerate(done_flags) if not d]

        is_heuristic = (
            opponent_model is not None
            and callable(opponent_model)
            and not isinstance(opponent_model, torch.nn.Module)
        )

        if opponent_model is not None:
            p1_active = [i for i in active if envs[i].current_player == 1]
            p2_active = [i for i in active if envs[i].current_player == 2]
            nn_groups = [(p1_active, model)]
            if not is_heuristic:
                nn_groups.append((p2_active, opponent_model))
        else:
            p2_active = []
            nn_groups = [(active, model)]

        # Map from env index to (action, log_prob, value, mask)
        step_results = {}

        for group_indices, m in nn_groups:
            if not group_indices:
                continue
            batch_states = torch.FloatTensor(
                np.array([states[i] for i in group_indices])
            ).to(device)
            batch_masks = torch.BoolTensor(
                np.array([envs[i].get_action_mask() for i in group_indices])
            ).to(device)

            with torch.no_grad():
                probs, values = m(batch_states, batch_masks)
                dist = torch.distributions.Categorical(probs)
                action_tensors = dist.sample()
                log_probs = dist.log_prob(action_tensors)

            for j, i in enumerate(group_indices):
                step_results[i] = (
                    action_tensors[j],
                    log_probs[j],
                    values[j],
                    batch_masks[j],
                )

        # Heuristic P2: call directly, no batching needed
        if is_heuristic:
            for i in p2_active:
                action = opponent_model(envs[i])
                mask_t = torch.BoolTensor(envs[i].get_action_mask()).to(device)
                step_results[i] = (
                    torch.tensor(action),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    mask_t,
                )

        for i in active:
            action_t, log_prob_t, value_t, mask_t = step_results[i]
            player = envs[i].current_player
            action = action_t.item()

            # Measure opponent threat before their move (only when heuristic mode)
            opp_threat_before = 0.0
            if opponent_model is not None and player == 2:
                opp_threat_before = envs[i]._evaluate_board(2)

            next_state, reward, done, info = envs[i].step(action)

            # After opponent's move: accumulate threat-growth penalty for P1
            if opponent_model is not None and player == 2:
                opp_threat_after = envs[i]._evaluate_board(2)
                delta = max(0.0, opp_threat_after - opp_threat_before)
                pending_threat_penalty[i] -= THREAT_PENALTY_COEF * delta

            # Only record steps for the training model (P1 when pool is active,
            # or all steps in standard self-play)
            if opponent_model is None or player == 1:
                adjusted_reward = reward + pending_threat_penalty[i]
                pending_threat_penalty[i] = 0.0
                episodes[i].append({
                    'state': states[i],
                    'action_mask': mask_t.cpu().numpy(),
                    'action': action,
                    'log_prob': log_prob_t.item(),
                    'value': value_t.item(),
                    'reward': adjusted_reward,
                    'done': done,
                    'player': player,
                })

            if done:
                done_flags[i] = True
                pending_threat_penalty[i] = 0.0
                if envs[i].winner is not None:
                    loser = 3 - envs[i].winner
                    for k in range(len(episodes[i]) - 1, -1, -1):
                        if episodes[i][k]['player'] == loser:
                            episodes[i][k]['reward'] = -1.0
                            break
            else:
                states[i] = next_state

    return episodes


def build_buffer(episodes: List[List[Dict]]) -> Dict:
    """Flatten episodes into a PPO buffer with computed GAE advantages.

    For self-play episodes (both players stored), GAE is computed per-player
    to avoid mixing value estimates across opposing perspectives. Player 1's
    'next value' bootstraps from Player 1's next turn, not Player 2's state.
    """
    buffer: Dict = {
        'states': [], 'action_masks': [], 'actions': [],
        'log_probs': [], 'returns': [], 'advantages': [],
    }

    for episode in episodes:
        players_in_ep = set(s['player'] for s in episode)

        if len(players_in_ep) == 1:
            # Single-player trajectory (vs heuristic / pool) — no interleaving issue
            trajectories = [episode]
        else:
            # Self-play: split by player so GAE never crosses opposing perspectives
            trajectories = [
                [s for s in episode if s['player'] == p]
                for p in sorted(players_in_ep)
            ]

        for traj in trajectories:
            if not traj:
                continue
            rewards = [s['reward'] for s in traj]
            values  = [s['value']  for s in traj]
            dones   = [s['done']   for s in traj]
            # Ensure the last step is terminal so GAE doesn't bootstrap past game end
            # (needed when the other player won, leaving this player's last done=False)
            dones[-1] = True

            advantages, returns = compute_gae(rewards, values, dones)

            buffer['states'].extend(s['state'] for s in traj)
            buffer['action_masks'].extend(s['action_mask'] for s in traj)
            buffer['actions'].extend(s['action'] for s in traj)
            buffer['log_probs'].extend(s['log_prob'] for s in traj)
            buffer['returns'].extend(returns)
            buffer['advantages'].extend(advantages)

    return buffer
