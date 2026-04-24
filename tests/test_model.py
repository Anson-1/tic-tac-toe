import torch
from super_tictactoe.model import ActorCritic

def test_output_shapes():
    model = ActorCritic()
    state = torch.zeros(1, 3, 12, 12)
    mask = torch.ones(1, 144, dtype=torch.bool)
    probs, value = model(state, mask)
    assert probs.shape == (1, 144)
    assert value.shape == (1, 1)

def test_probs_sum_to_one():
    model = ActorCritic()
    state = torch.zeros(1, 3, 12, 12)
    mask = torch.ones(1, 144, dtype=torch.bool)
    probs, _ = model(state, mask)
    assert abs(probs.sum().item() - 1.0) < 1e-5

def test_masked_actions_have_zero_prob():
    model = ActorCritic()
    state = torch.zeros(1, 3, 12, 12)
    mask = torch.ones(1, 144, dtype=torch.bool)
    mask[0, :10] = False  # mask first 10 actions
    probs, _ = model(state, mask)
    assert (probs[0, :10] == 0).all()
    assert (probs[0, 10:] > 0).all()

def test_get_action_returns_valid_action():
    model = ActorCritic()
    state = torch.zeros(3, 12, 12)
    mask = torch.zeros(144, dtype=torch.bool)
    mask[50:60] = True  # only actions 50-59 valid
    action, log_prob, value = model.get_action(state, mask)
    assert 50 <= action < 60
    assert log_prob.shape == torch.Size([])
    assert value.shape == torch.Size([])

def test_forward_batch():
    model = ActorCritic()
    states = torch.zeros(8, 3, 12, 12)
    masks = torch.ones(8, 144, dtype=torch.bool)
    probs, values = model(states, masks)
    assert probs.shape == (8, 144)
    assert values.shape == (8, 1)
