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
    n_envs: int, model: ActorCritic, device: str = 'cpu'
) -> List[List[Dict]]:
    """
    Run n_envs games simultaneously with one batched forward pass per step.
    Returns list of n_envs episodes, each a list of step dicts.
    """
    envs = [SuperTicTacToeEnv() for _ in range(n_envs)]
    states = [env.reset() for env in envs]
    episodes: List[List[Dict]] = [[] for _ in range(n_envs)]
    done_flags = [False] * n_envs

    while not all(done_flags):
        active = [i for i, d in enumerate(done_flags) if not d]

        batch_states = torch.FloatTensor(
            np.array([states[i] for i in active])
        ).to(device)
        batch_masks = torch.BoolTensor(
            np.array([envs[i].get_action_mask() for i in active])
        ).to(device)

        with torch.no_grad():
            probs, values = model(batch_states, batch_masks)
            dist = torch.distributions.Categorical(probs)
            action_tensors = dist.sample()
            log_probs = dist.log_prob(action_tensors)

        for j, i in enumerate(active):
            player = envs[i].current_player
            action = action_tensors[j].item()

            next_state, reward, done, info = envs[i].step(action)

            episodes[i].append({
                'state': states[i],
                'action_mask': batch_masks[j].cpu().numpy(),
                'action': action,
                'log_prob': log_probs[j].item(),
                'value': values[j].item(),
                'reward': reward,
                'done': done,
                'player': player,
            })

            if done:
                done_flags[i] = True
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
    """Flatten episodes into a PPO buffer with computed GAE advantages."""
    buffer: Dict = {
        'states': [], 'action_masks': [], 'actions': [],
        'log_probs': [], 'returns': [], 'advantages': [],
    }

    for episode in episodes:
        rewards = [s['reward'] for s in episode]
        values = [s['value'] for s in episode]
        dones = [s['done'] for s in episode]
        advantages, returns = compute_gae(rewards, values, dones)

        buffer['states'].extend(s['state'] for s in episode)
        buffer['action_masks'].extend(s['action_mask'] for s in episode)
        buffer['actions'].extend(s['action'] for s in episode)
        buffer['log_probs'].extend(s['log_prob'] for s in episode)
        buffer['returns'].extend(returns)
        buffer['advantages'].extend(advantages)

    return buffer
