import torch
import numpy as np
from super_tictactoe.ppo import compute_gae, ppo_update
from super_tictactoe.model import ActorCritic


def test_gae_terminal_reward():
    """Single-step episode: advantage = reward - value."""
    rewards = [1.0]
    values = [0.4]
    dones = [True]
    advantages, returns = compute_gae(rewards, values, dones)
    assert abs(returns[0] - 1.0) < 1e-5
    assert abs(advantages[0] - (1.0 - 0.4)) < 1e-4

def test_gae_discounting():
    """Two-step episode: return at t=0 = r0 + gamma*r1."""
    rewards = [0.0, 1.0]
    values = [0.5, 0.8]
    dones = [False, True]
    gamma = 0.99
    advantages, returns = compute_gae(rewards, values, dones, gamma=gamma)
    expected_return_0 = 0.0 + gamma * 1.0
    assert abs(returns[0] - expected_return_0) < 1e-4

def test_gae_lengths_match_input():
    rewards = [0.0, 0.0, 1.0]
    values = [0.3, 0.5, 0.9]
    dones = [False, False, True]
    advantages, returns = compute_gae(rewards, values, dones)
    assert len(advantages) == 3
    assert len(returns) == 3

def test_ppo_update_returns_losses():
    """ppo_update should return a dict with actor_loss and critic_loss."""
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    n = 16
    buffer = {
        'states': np.zeros((n, 3, 12, 12), dtype=np.float32),
        'action_masks': np.ones((n, 144), dtype=bool),
        'actions': np.zeros(n, dtype=np.int64),
        'log_probs': np.full(n, -4.0, dtype=np.float32),
        'returns': np.ones(n, dtype=np.float32),
        'advantages': np.ones(n, dtype=np.float32),
    }
    for i in range(n):
        buffer['actions'][i] = 50

    losses = ppo_update(model, optimizer, buffer, epochs=2)
    assert 'actor_loss' in losses
    assert 'critic_loss' in losses
    assert isinstance(losses['actor_loss'], float)
    assert isinstance(losses['critic_loss'], float)
