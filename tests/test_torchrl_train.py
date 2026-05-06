import torch
import pytest
from super_tictactoe.torchrl_env import SuperTicTacToeTorchEnv
from super_tictactoe.model import ActorCritic
import torchrl_train


def test_actor_output_keys():
    env = SuperTicTacToeTorchEnv()
    ac = ActorCritic()
    actor, critic = torchrl_train.make_modules(ac, env.action_spec)

    td = env.reset()
    actor(td)
    assert "action" in td.keys(), "actor must write 'action'"
    assert "sample_log_prob" in td.keys(), "actor must write 'sample_log_prob'"
    assert td["action"].shape == torch.Size([])
    assert int(td["action"].item()) in range(144)


def test_critic_output_keys():
    env = SuperTicTacToeTorchEnv()
    ac = ActorCritic()
    actor, critic = torchrl_train.make_modules(ac, env.action_spec)

    td = env.reset()
    critic(td)
    assert "state_value" in td.keys(), "critic must write 'state_value'"
    assert td["state_value"].shape == torch.Size([1])


def test_action_is_valid_cell():
    """Sampled actions must be within the valid mask."""
    env = SuperTicTacToeTorchEnv()
    ac = ActorCritic()
    actor, _ = torchrl_train.make_modules(ac, env.action_spec)

    for _ in range(20):
        td = env.reset()
        actor(td)
        action = int(td["action"].item())
        assert td["action_mask"][action].item(), f"Action {action} is masked-out"
