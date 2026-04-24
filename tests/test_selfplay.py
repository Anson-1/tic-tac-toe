import numpy as np
import torch
from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.model import ActorCritic
from super_tictactoe.selfplay import collect_episode, build_buffer, collect_episodes_vectorized


def test_collect_episode_returns_list():
    env = SuperTicTacToeEnv()
    model = ActorCritic()
    episode = collect_episode(env, model)
    assert isinstance(episode, list)
    assert len(episode) > 0

def test_collect_episode_keys():
    env = SuperTicTacToeEnv()
    model = ActorCritic()
    episode = collect_episode(env, model)
    step = episode[0]
    for key in ('state', 'action_mask', 'action', 'log_prob', 'value', 'reward', 'done', 'player'):
        assert key in step, f"Missing key: {key}"

def test_collect_episode_state_shape():
    env = SuperTicTacToeEnv()
    model = ActorCritic()
    episode = collect_episode(env, model)
    assert episode[0]['state'].shape == (3, 12, 12)

def test_collect_episode_winner_gets_positive_reward():
    env = SuperTicTacToeEnv()
    model = ActorCritic()
    episode = collect_episode(env, model)
    if env.winner is not None:
        winner_rewards = [s['reward'] for s in episode if s['player'] == env.winner]
        assert any(r > 0 for r in winner_rewards)

def test_collect_episode_loser_gets_negative_reward():
    env = SuperTicTacToeEnv()
    model = ActorCritic()
    episode = collect_episode(env, model)
    if env.winner is not None:
        loser = 3 - env.winner
        loser_rewards = [s['reward'] for s in episode if s['player'] == loser]
        assert any(r < 0 for r in loser_rewards)

def test_build_buffer_keys():
    env = SuperTicTacToeEnv()
    model = ActorCritic()
    episodes = [collect_episode(env, model) for _ in range(3)]
    buffer = build_buffer(episodes)
    for key in ('states', 'action_masks', 'actions', 'log_probs', 'returns', 'advantages'):
        assert key in buffer, f"Missing buffer key: {key}"

def test_build_buffer_lengths_match():
    env = SuperTicTacToeEnv()
    model = ActorCritic()
    episodes = [collect_episode(env, model) for _ in range(3)]
    total_steps = sum(len(ep) for ep in episodes)
    buffer = build_buffer(episodes)
    assert len(buffer['states']) == total_steps
    assert len(buffer['actions']) == total_steps


def test_vectorized_returns_n_episodes():
    model = ActorCritic()
    episodes = collect_episodes_vectorized(4, model)
    assert len(episodes) == 4

def test_vectorized_episode_keys():
    model = ActorCritic()
    episodes = collect_episodes_vectorized(2, model)
    step = episodes[0][0]
    for key in ('state', 'action_mask', 'action', 'log_prob', 'value', 'reward', 'done', 'player'):
        assert key in step, f"Missing key: {key}"

def test_vectorized_state_shape():
    model = ActorCritic()
    episodes = collect_episodes_vectorized(2, model)
    assert episodes[0][0]['state'].shape == (3, 12, 12)

def test_vectorized_winner_gets_positive_reward():
    model = ActorCritic()
    episodes = collect_episodes_vectorized(4, model)
    for ep in episodes:
        env_winner = None
        for step in ep:
            if step['reward'] == 1.0:
                env_winner = step['player']
        if env_winner is not None:
            winner_rewards = [s['reward'] for s in ep if s['player'] == env_winner]
            assert any(r > 0 for r in winner_rewards)

def test_vectorized_loser_gets_negative_reward():
    model = ActorCritic()
    episodes = collect_episodes_vectorized(4, model)
    for ep in episodes:
        env_winner = None
        for step in ep:
            if step['reward'] == 1.0:
                env_winner = step['player']
        if env_winner is not None:
            loser = 3 - env_winner
            loser_rewards = [s['reward'] for s in ep if s['player'] == loser]
            assert any(r < 0 for r in loser_rewards)
