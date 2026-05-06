import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    Composite,
    Unbounded,
    Categorical,
    Binary,
)

from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.heuristics import random_heuristic


class SuperTicTacToeTorchEnv(EnvBase):
    """
    TorchRL EnvBase wrapper for Super Tic-Tac-Toe.

    Presents the two-player game as single-agent: after the agent (P1) acts,
    the opponent (P2/heuristic) plays immediately inside _step(). The agent
    always plays as P1. The active opponent is a swappable attribute so the
    training loop can implement curriculum by writing env.opponent_fn = new_fn.
    """

    def __init__(
        self,
        opponent_fn=None,
        success_rate: float = 0.5,
        device: str = "cpu",
        seed: int = 0,
    ):
        super().__init__(device=device, batch_size=[])
        self._env = SuperTicTacToeEnv(success_rate=success_rate)
        self.opponent_fn = opponent_fn if opponent_fn is not None else random_heuristic
        self._make_spec()
        self._set_seed(seed)

    def _make_spec(self):
        self.observation_spec = Composite(
            observation=Unbounded(
                shape=(3, 12, 12), dtype=torch.float32, device=self.device
            ),
            action_mask=Binary(
                n=144, dtype=torch.bool, device=self.device
            ),
            shape=(),
            device=self.device,
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = Categorical(
            n=144, shape=(), dtype=torch.int64, device=self.device
        )
        self.reward_spec = Unbounded(
            shape=(1,), dtype=torch.float32, device=self.device
        )

    def _reset(self, tensordict=None):
        obs = self._env.reset()
        mask = self._env.get_action_mask()
        return TensorDict(
            {
                "observation": torch.from_numpy(obs.copy()).float(),
                "action_mask": torch.from_numpy(mask.copy()),
            },
            batch_size=[],
            device=self.device,
        )

    def _step(self, tensordict):
        action = int(tensordict["action"].item())

        # Agent (P1) acts
        next_obs, reward, done, _ = self._env.step(action)

        # Opponent (P2) acts immediately if game is not over
        if not done:
            opp_action = self.opponent_fn(self._env)
            next_obs, _, done, _ = self._env.step(opp_action)
            # If opponent won, agent receives the loss penalty
            if done and self._env.winner == 2:
                reward = -1.0

        mask = self._env.get_action_mask()
        return TensorDict(
            {
                "observation": torch.from_numpy(next_obs.copy()).float(),
                "action_mask": torch.from_numpy(mask.copy()),
                "reward": torch.tensor([float(reward)], dtype=torch.float32),
                "done": torch.tensor([done]),
                "terminated": torch.tensor([done]),
            },
            batch_size=[],
            device=self.device,
        )

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
