import pytest
import torch
from super_tictactoe.torchrl_env import SuperTicTacToeTorchEnv


def make_env():
    return SuperTicTacToeTorchEnv(seed=42)


def test_observation_spec_shapes():
    env = make_env()
    assert tuple(env.observation_spec["observation"].shape) == (3, 12, 12)
    assert tuple(env.observation_spec["action_mask"].shape) == (144,)


def test_reset_keys_and_shapes():
    env = make_env()
    td = env.reset()
    assert "observation" in td.keys()
    assert "action_mask" in td.keys()
    assert td["observation"].shape == torch.Size([3, 12, 12])
    assert td["action_mask"].shape == torch.Size([144])
    assert td["observation"].dtype == torch.float32
    assert td["action_mask"].dtype == torch.bool


def test_step_keys():
    env = make_env()
    td = env.reset()
    valid = torch.where(td["action_mask"])[0]
    td["action"] = valid[0]
    result = env._step(td)
    for key in ["observation", "action_mask", "reward", "done", "terminated"]:
        assert key in result.keys(), f"Missing key: {key}"
    assert result["reward"].shape == torch.Size([1])
    assert result["done"].shape == torch.Size([1])


def test_episode_terminates():
    env = make_env()
    td = env.reset()
    for _ in range(300):
        valid = torch.where(td["action_mask"])[0]
        if len(valid) == 0:
            break
        td["action"] = valid[torch.randint(len(valid), (1,))]
        result = env._step(td)
        if result["done"].item():
            return  # success
        td = result
    pytest.fail("Episode did not terminate within 300 steps")


def test_opponent_is_swappable():
    from super_tictactoe.heuristics import greedy_agent
    env = make_env()
    env.opponent_fn = greedy_agent
    td = env.reset()
    valid = torch.where(td["action_mask"])[0]
    td["action"] = valid[0]
    env._step(td)  # must not raise
