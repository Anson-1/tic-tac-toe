# Super Tic-Tac-Toe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Super Tic-Tac-Toe game with a PPO self-play agent and Pygame GUI supporting human vs agent and agent vs agent modes.

**Architecture:** Game environment uses a 12×12 grid (6 sub-grids of 4×4) with perspective-flipped 3-channel state. CNN actor-critic trained via PPO self-play. Pygame GUI renders the centered triangle board.

**Tech Stack:** Python 3.11, PyTorch, Pygame, NumPy, pytest

---

## File Map

```
super_tictactoe/
├── __init__.py
├── env.py         ← board state, stochastic placement, win detection, step/reset
├── model.py       ← CNN actor-critic (shared backbone, actor + critic heads)
├── ppo.py         ← GAE computation, PPO clip update
├── selfplay.py    ← episode collection, buffer construction
├── train.py       ← training loop, checkpointing
├── evaluate.py    ← agent vs agent win rate evaluation
└── gui.py         ← Pygame: human vs agent, agent vs agent
tests/
├── __init__.py
├── test_env.py
├── test_model.py
├── test_ppo.py
└── test_selfplay.py
requirements.txt
pytest.ini
```

---

### Task 1: Project Setup

**Files:**
- Create: `super_tictactoe/__init__.py`
- Create: `tests/__init__.py`
- Create: `requirements.txt`
- Create: `pytest.ini`

- [ ] **Step 1: Create project structure**

```bash
mkdir -p super_tictactoe tests checkpoints
touch super_tictactoe/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `requirements.txt`**

```
torch>=2.0.0
pygame>=2.5.0
numpy>=1.24.0
pytest>=7.0.0
```

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
```

- [ ] **Step 4: Verify pytest runs**

```bash
pytest
```
Expected: `no tests ran`

- [ ] **Step 5: Commit**

```bash
git init
git add .
git commit -m "chore: project setup"
```

---

### Task 2: Board Representation and Valid Mask

**Files:**
- Create: `super_tictactoe/env.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_env.py
import numpy as np
from super_tictactoe.env import GRID_POSITIONS, SuperTicTacToeEnv

def test_valid_mask_has_96_cells():
    env = SuperTicTacToeEnv()
    assert env.valid_mask.sum() == 96

def test_valid_mask_shape():
    env = SuperTicTacToeEnv()
    assert env.valid_mask.shape == (12, 12)

def test_grid_positions_count():
    assert len(GRID_POSITIONS) == 6

def test_level1_grid_centered():
    env = SuperTicTacToeEnv()
    assert env.valid_mask[0:4, 4:8].all()
    assert not env.valid_mask[0:4, 0:4].any()
    assert not env.valid_mask[0:4, 8:12].any()

def test_level2_grids_centered():
    env = SuperTicTacToeEnv()
    assert env.valid_mask[4:8, 2:6].all()
    assert env.valid_mask[4:8, 6:10].all()
    assert not env.valid_mask[4:8, 0:2].any()
    assert not env.valid_mask[4:8, 10:12].any()

def test_level3_full_width():
    env = SuperTicTacToeEnv()
    assert env.valid_mask[8:12, 0:12].all()

def test_reset_state_shape():
    env = SuperTicTacToeEnv()
    state = env.reset()
    assert state.shape == (3, 12, 12)

def test_reset_board_empty():
    env = SuperTicTacToeEnv()
    env.reset()
    assert (env.board == 0).all()

def test_reset_player_is_1():
    env = SuperTicTacToeEnv()
    env.reset()
    assert env.current_player == 1

def test_action_mask_shape():
    env = SuperTicTacToeEnv()
    env.reset()
    mask = env.get_action_mask()
    assert mask.shape == (144,)
    assert mask.sum() == 96
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_env.py -v
```
Expected: `ModuleNotFoundError: No module named 'super_tictactoe.env'`

- [ ] **Step 3: Write minimal implementation**

```python
# super_tictactoe/env.py
import numpy as np
from typing import Optional, Tuple

# (row_start, col_start) in 12×12 grid for each sub-grid
GRID_POSITIONS = [
    (0, 4),   # G0: Level 1 (centered)
    (4, 2),   # G1: Level 2 left
    (4, 6),   # G2: Level 2 right
    (8, 0),   # G3: Level 3 left
    (8, 4),   # G4: Level 3 center
    (8, 8),   # G5: Level 3 right
]

# Row ranges [inclusive] per level
LEVEL_ROW_RANGES = [(0, 3), (4, 7), (8, 11)]


class SuperTicTacToeEnv:
    def __init__(self):
        self.valid_mask = self._build_valid_mask()
        self.board = np.zeros((12, 12), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None

    def _build_valid_mask(self) -> np.ndarray:
        mask = np.zeros((12, 12), dtype=bool)
        for r, c in GRID_POSITIONS:
            mask[r:r+4, c:c+4] = True
        return mask

    def reset(self) -> np.ndarray:
        self.board = np.zeros((12, 12), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """(3, 12, 12) state from current player's perspective."""
        state = np.zeros((3, 12, 12), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == 3 - self.current_player).astype(np.float32)
        state[2] = (self.valid_mask & (self.board == 0)).astype(np.float32)
        return state

    def get_action_mask(self) -> np.ndarray:
        """(144,) bool: True = valid and empty cell."""
        return (self.valid_mask & (self.board == 0)).flatten()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_env.py -v
```
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add super_tictactoe/env.py tests/test_env.py
git commit -m "feat: board representation and valid mask"
```

---

### Task 3: Stochastic Placement

**Files:**
- Modify: `super_tictactoe/env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_env.py

def test_stochastic_chosen_cell_50pct():
    """~50% of placements land on chosen cell."""
    env = SuperTicTacToeEnv()
    env.reset()
    hits = sum(1 for _ in range(2000) if env._stochastic_place(5, 4) == (5, 4))
    assert 850 < hits < 1150  # 1000 expected ± 150

def test_stochastic_corner_forfeit_rate():
    """Corner (0,4) of G0: P(forfeit) = 1/2 * 5/8 = 5/16 ≈ 0.3125."""
    env = SuperTicTacToeEnv()
    env.reset()
    forfeits = sum(1 for _ in range(3200) if env._stochastic_place(0, 4) == (None, None))
    assert 800 < forfeits < 1200  # 1000 expected ± 200

def test_stochastic_occupied_neighbor_forfeits():
    """All valid neighbors occupied → 50% chance: chosen cell, 50% chance: forfeit."""
    env = SuperTicTacToeEnv()
    env.reset()
    # (0,4) is top-left corner of G0. Valid neighbors within G0: (0,5),(1,4),(1,5)
    env.board[0, 5] = 1
    env.board[1, 4] = 1
    env.board[1, 5] = 1
    results = [env._stochastic_place(0, 4) for _ in range(1000)]
    assert all(r in [(0, 4), (None, None)] for r in results)

def test_stochastic_returns_valid_cell():
    """Non-corner placement never goes to padding area."""
    env = SuperTicTacToeEnv()
    env.reset()
    for _ in range(500):
        result = env._stochastic_place(5, 4)  # center of G1
        if result != (None, None):
            r, c = result
            assert env.valid_mask[r, c], f"Placed at invalid cell ({r},{c})"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_env.py::test_stochastic_chosen_cell_50pct -v
```
Expected: `AttributeError: 'SuperTicTacToeEnv' object has no attribute '_stochastic_place'`

- [ ] **Step 3: Write minimal implementation**

```python
# Add to SuperTicTacToeEnv class body in super_tictactoe/env.py
# _DIRECTIONS is a class-level attribute (indented inside the class, before methods)

    _DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def _get_grid(self, row: int, col: int) -> Optional[int]:
    for i, (r, c) in enumerate(GRID_POSITIONS):
        if r <= row < r + 4 and c <= col < c + 4:
            return i
    return None

def _stochastic_place(self, row: int, col: int) -> Tuple[Optional[int], Optional[int]]:
    """
    50%: returns (row, col).
    50%: picks one of 8 adjacent cells uniformly.
         Returns (None, None) if out of sub-grid bounds or occupied.
    """
    if np.random.random() < 0.5:
        return row, col

    dr, dc = self._DIRECTIONS[np.random.randint(8)]
    new_row, new_col = row + dr, col + dc

    grid_id = self._get_grid(row, col)
    r_start, c_start = GRID_POSITIONS[grid_id]
    if not (r_start <= new_row < r_start + 4 and c_start <= new_col < c_start + 4):
        return None, None
    if self.board[new_row, new_col] != 0:
        return None, None
    return new_row, new_col
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_env.py::test_stochastic_chosen_cell_50pct tests/test_env.py::test_stochastic_corner_forfeit_rate tests/test_env.py::test_stochastic_occupied_neighbor_forfeits tests/test_env.py::test_stochastic_returns_valid_cell -v
```
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add super_tictactoe/env.py tests/test_env.py
git commit -m "feat: stochastic placement"
```

---

### Task 4: Win Detection

**Files:**
- Modify: `super_tictactoe/env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_env.py

def _place(env, player, cells):
    """Helper: directly set cells on board for a given player."""
    for r, c in cells:
        env.board[r, c] = player

def test_horizontal_win_4_in_row():
    env = SuperTicTacToeEnv()
    env.reset()
    # G0 (rows 0-3, cols 4-7): 4 in row 0
    _place(env, 1, [(0,4),(0,5),(0,6),(0,7)])
    assert env._check_win(1)

def test_horizontal_no_win_3_in_row():
    env = SuperTicTacToeEnv()
    env.reset()
    _place(env, 1, [(0,4),(0,5),(0,6)])
    assert not env._check_win(1)

def test_vertical_win_cross_level():
    env = SuperTicTacToeEnv()
    env.reset()
    # Column 4: rows 2,3 (Level 1), rows 4,5 (Level 2) — spans 2 levels
    _place(env, 1, [(2,4),(3,4),(4,4),(5,4)])
    assert env._check_win(1)

def test_vertical_no_win_same_level():
    env = SuperTicTacToeEnv()
    env.reset()
    # Column 4: rows 8,9,10,11 — all Level 3, no cross-level
    _place(env, 1, [(8,4),(9,4),(10,4),(11,4)])
    assert not env._check_win(1)

def test_vertical_no_win_3_cross_level():
    env = SuperTicTacToeEnv()
    env.reset()
    # Only 3 cells, even if cross-level
    _place(env, 1, [(3,4),(4,4),(5,4)])
    assert not env._check_win(1)

def test_diagonal_win_5_topleft_to_bottomright():
    env = SuperTicTacToeEnv()
    env.reset()
    # Diagonal from (3,4) going down-right: (3,4),(4,5),(5,6),(6,7),(7,8)
    # Check validity: all must be in valid_mask
    cells = [(3,4),(4,5),(5,6),(6,7),(7,8)]
    assert all(env.valid_mask[r,c] for r,c in cells), "Test cells not all valid"
    _place(env, 1, cells)
    assert env._check_win(1)

def test_diagonal_win_5_topright_to_bottomleft():
    env = SuperTicTacToeEnv()
    env.reset()
    # Diagonal going down-left: (3,7),(4,6),(5,5),(6,4),(7,3)
    cells = [(3,7),(4,6),(5,5),(6,4),(7,3)]
    assert all(env.valid_mask[r,c] for r,c in cells), "Test cells not all valid"
    _place(env, 1, cells)
    assert env._check_win(1)

def test_diagonal_no_win_4():
    env = SuperTicTacToeEnv()
    env.reset()
    cells = [(3,4),(4,5),(5,6),(6,7)]
    _place(env, 1, cells)
    assert not env._check_win(1)

def test_no_win_empty_board():
    env = SuperTicTacToeEnv()
    env.reset()
    assert not env._check_win(1)
    assert not env._check_win(2)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_env.py -k "win" -v
```
Expected: `AttributeError: 'SuperTicTacToeEnv' object has no attribute '_check_win'`

- [ ] **Step 3: Write minimal implementation**

```python
# Add to SuperTicTacToeEnv in super_tictactoe/env.py

def _get_level(self, row: int) -> int:
    if row <= 3: return 0
    if row <= 7: return 1
    return 2

def _check_win(self, player: int) -> bool:
    b = (self.board == player)

    # Horizontal: 4 consecutive valid cells in same row
    for r in range(12):
        for c in range(9):  # c+3 <= 11
            if all(self.valid_mask[r, c+i] and b[r, c+i] for i in range(4)):
                return True

    # Vertical: 4 consecutive valid cells in same column, spanning 2+ levels
    for c in range(12):
        for r in range(9):  # r+3 <= 11
            cells = [(r+i, c) for i in range(4)]
            if all(self.valid_mask[rr, cc] and b[rr, cc] for rr, cc in cells):
                levels = {self._get_level(rr) for rr, _ in cells}
                if len(levels) >= 2:
                    return True

    # Diagonal ↘: 5 consecutive valid cells
    for r in range(8):  # r+4 <= 11
        for c in range(8):  # c+4 <= 11
            if all(self.valid_mask[r+i, c+i] and b[r+i, c+i] for i in range(5)):
                return True

    # Diagonal ↙: 5 consecutive valid cells
    for r in range(8):
        for c in range(4, 12):  # c-4 >= 0
            if all(self.valid_mask[r+i, c-i] and b[r+i, c-i] for i in range(5)):
                return True

    return False
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_env.py -k "win" -v
```
Expected: All 9 PASS

- [ ] **Step 5: Commit**

```bash
git add super_tictactoe/env.py tests/test_env.py
git commit -m "feat: win detection (horizontal, vertical cross-level, diagonal)"
```

---

### Task 5: Full Game Environment (step)

**Files:**
- Modify: `super_tictactoe/env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to tests/test_env.py (also add at top of file: import unittest.mock)

def test_step_places_piece_on_board():
    env = SuperTicTacToeEnv()
    env.reset()
    np.random.seed(0)  # deterministic for this test
    # Force placement at chosen cell by seeding
    action = 4 * 12 + 2  # row=4, col=2 (G1 top-left)
    with unittest.mock.patch('numpy.random.random', return_value=0.1):  # < 0.5 → chosen cell
        state, reward, done, info = env.step(action)
    assert env.board[4, 2] != 0 or info['forfeited']

def test_step_switches_player():
    env = SuperTicTacToeEnv()
    env.reset()
    assert env.current_player == 1
    with unittest.mock.patch('numpy.random.random', return_value=0.1):
        env.step(4 * 12 + 2)
    assert env.current_player == 2

def test_step_returns_correct_shapes():
    env = SuperTicTacToeEnv()
    env.reset()
    state, reward, done, info = env.step(4 * 12 + 2)
    assert state.shape == (3, 12, 12)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

def test_step_win_terminates():
    env = SuperTicTacToeEnv()
    env.reset()
    # Place 3 pieces for P1, then win with step()
    env.board[0, 4] = 1; env.board[0, 5] = 1; env.board[0, 6] = 1
    env.current_player = 1
    action = 0 * 12 + 7  # (0, 7) → completes 4-in-row in G0
    with unittest.mock.patch('numpy.random.random', return_value=0.1):
        state, reward, done, info = env.step(action)
    assert reward == 1.0
    assert done
    assert env.winner == 1

def test_step_forfeit_no_piece_placed():
    env = SuperTicTacToeEnv()
    env.reset()
    # Force forfeit: 50% branch, pick direction out of bounds at (0,4)
    with unittest.mock.patch('numpy.random.random', return_value=0.6), \
         unittest.mock.patch('numpy.random.randint', return_value=0):  # direction (-1,-1)
        state, reward, done, info = env.step(0 * 12 + 4)
    assert info['forfeited']
    assert env.board[0, 4] == 0  # nothing placed

def test_step_draw_when_board_full():
    env = SuperTicTacToeEnv()
    env.reset()
    # Fill all valid cells except one with alternating pieces
    cells = list(zip(*np.where(env.valid_mask)))
    for i, (r, c) in enumerate(cells[:-1]):
        env.board[r, c] = 1 if i % 2 == 0 else 2
    env.current_player = 1
    last_r, last_c = cells[-1]
    action = last_r * 12 + last_c
    with unittest.mock.patch('numpy.random.random', return_value=0.1):
        _, reward, done, _ = env.step(action)
    assert done
    assert env.winner is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_env.py -k "step" -v
```
Expected: `AttributeError: 'SuperTicTacToeEnv' object has no attribute 'step'`

- [ ] **Step 3: Write minimal implementation**

```python
# Add to SuperTicTacToeEnv in super_tictactoe/env.py
# Add at top of file: import unittest.mock (not needed in env.py — tests use it)

def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
    """
    action: int in [0, 143] (row * 12 + col in 12×12 grid)
    Returns: (next_state, reward, done, info)
    reward is from the perspective of the player who just moved.
    """
    assert not self.done, "Cannot call step() on a finished game"
    row, col = action // 12, action % 12
    assert self.valid_mask[row, col], f"Action {action} targets padding cell ({row},{col})"
    assert self.board[row, col] == 0, f"Cell ({row},{col}) is already occupied"

    placed = self._stochastic_place(row, col)
    forfeited = placed == (None, None)

    if not forfeited:
        placed_row, placed_col = placed
        self.board[placed_row, placed_col] = self.current_player

    reward = 0.0
    if not forfeited and self._check_win(self.current_player):
        reward = 1.0
        self.done = True
        self.winner = self.current_player
    elif np.all(self.board[self.valid_mask] != 0):
        self.done = True

    info = {
        'forfeited': forfeited,
        'placed': placed if not forfeited else None,
        'player': self.current_player,
    }
    self.current_player = 3 - self.current_player
    return self._get_state(), reward, self.done, info
```

- [ ] **Step 4: Run all env tests**

```bash
pytest tests/test_env.py -v
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add super_tictactoe/env.py tests/test_env.py
git commit -m "feat: full game environment step() with forfeit and draw handling"
```

---

### Task 6: Neural Network (CNN Actor-Critic)

**Files:**
- Create: `super_tictactoe/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
import torch
import numpy as np
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_model.py -v
```
Expected: `ModuleNotFoundError: No module named 'super_tictactoe.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# super_tictactoe/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 12 * 12, 256)  # 9216 → 256
        self.actor = nn.Linear(256, 144)
        self.critic = nn.Linear(256, 1)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

    def forward(
        self, x: torch.Tensor, action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, 3, 12, 12)
        action_mask: (batch, 144) bool — True = valid action
        Returns: probs (batch, 144), value (batch, 1)
        """
        features = self._backbone(x)
        logits = self.actor(features)
        logits = logits.masked_fill(~action_mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        value = self.critic(features)
        return probs, value

    def get_action(
        self, state: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        state: (3, 12, 12) — single state, no batch dim
        action_mask: (144,) bool
        Returns: action (int), log_prob (scalar tensor), value (scalar tensor)
        """
        probs, value = self.forward(state.unsqueeze(0), action_mask.unsqueeze(0))
        probs = probs.squeeze(0)
        if deterministic:
            action = probs.argmax().item()
            log_prob = torch.log(probs[action] + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            action = action_tensor.item()
        return action, log_prob, value.squeeze()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_model.py -v
```
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add super_tictactoe/model.py tests/test_model.py
git commit -m "feat: CNN actor-critic with action masking"
```

---

### Task 7: PPO Update (GAE + Losses)

**Files:**
- Create: `super_tictactoe/ppo.py`
- Create: `tests/test_ppo.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ppo.py
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
    # Set one valid action per state (action 50)
    for i in range(n):
        buffer['actions'][i] = 50

    losses = ppo_update(model, optimizer, buffer, epochs=2)
    assert 'actor_loss' in losses
    assert 'critic_loss' in losses
    assert isinstance(losses['actor_loss'], float)
    assert isinstance(losses['critic_loss'], float)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_ppo.py -v
```
Expected: `ModuleNotFoundError: No module named 'super_tictactoe.ppo'`

- [ ] **Step 3: Write minimal implementation**

```python
# super_tictactoe/ppo.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE).
    Returns (advantages, returns).
    """
    advantages = []
    gae = 0.0
    next_value = 0.0

    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * next_value * (1.0 - float(d)) - v
        gae = delta + gamma * lam * (1.0 - float(d)) * gae
        advantages.insert(0, gae)
        next_value = v

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def ppo_update(
    model,
    optimizer: torch.optim.Optimizer,
    buffer: Dict,
    epochs: int = 4,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> Dict[str, float]:
    """Run PPO update over the collected buffer."""
    states = torch.FloatTensor(np.array(buffer['states']))
    action_masks = torch.BoolTensor(np.array(buffer['action_masks']))
    actions = torch.LongTensor(np.array(buffer['actions']))
    old_log_probs = torch.FloatTensor(np.array(buffer['log_probs']))
    returns = torch.FloatTensor(np.array(buffer['returns']))
    advantages = torch.FloatTensor(np.array(buffer['advantages']))

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_losses, critic_losses = [], []

    for _ in range(epochs):
        probs, values = model(states, action_masks)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(values.squeeze(), returns)
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

    return {
        'actor_loss': float(np.mean(actor_losses)),
        'critic_loss': float(np.mean(critic_losses)),
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_ppo.py -v
```
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add super_tictactoe/ppo.py tests/test_ppo.py
git commit -m "feat: PPO update with GAE"
```

---

### Task 8: Self-Play Episode Collection

**Files:**
- Create: `super_tictactoe/selfplay.py`
- Create: `tests/test_selfplay.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_selfplay.py
import numpy as np
import torch
from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.model import ActorCritic
from super_tictactoe.selfplay import collect_episode, build_buffer


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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_selfplay.py -v
```
Expected: `ModuleNotFoundError: No module named 'super_tictactoe.selfplay'`

- [ ] **Step 3: Write minimal implementation**

```python
# super_tictactoe/selfplay.py
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
        action_mask = torch.BoolTensor(env.get_action_mask())
        state_tensor = torch.FloatTensor(state).to(device)

        with torch.no_grad():
            action, log_prob, value = model.get_action(state_tensor, action_mask)

        next_state, reward, done, info = env.step(action)

        episode.append({
            'state': state,
            'action_mask': action_mask.numpy(),
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_selfplay.py -v
```
Expected: All 7 PASS

- [ ] **Step 5: Commit**

```bash
git add super_tictactoe/selfplay.py tests/test_selfplay.py
git commit -m "feat: self-play episode collection and PPO buffer"
```

---

### Task 9: Training Script

**Files:**
- Create: `super_tictactoe/train.py`

- [ ] **Step 1: Write the training script**

```python
# super_tictactoe/train.py
import os
import torch
import argparse
from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.model import ActorCritic
from super_tictactoe.selfplay import collect_episode, build_buffer
from super_tictactoe.ppo import ppo_update


def train(
    num_updates: int = 1000,
    episodes_per_update: int = 512,
    save_every: int = 100,
    device: str = 'cpu',
    checkpoint_dir: str = 'checkpoints',
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    env = SuperTicTacToeEnv()
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for update in range(1, num_updates + 1):
        episodes = [collect_episode(env, model, device) for _ in range(episodes_per_update)]
        buffer = build_buffer(episodes)
        losses = ppo_update(model, optimizer, buffer)

        wins = sum(1 for ep in episodes if env.winner is not None)
        if update % 10 == 0:
            print(
                f"Update {update:4d} | "
                f"actor_loss={losses['actor_loss']:.4f} | "
                f"critic_loss={losses['critic_loss']:.4f}"
            )

        if update % save_every == 0:
            path = os.path.join(checkpoint_dir, f"model_{update:04d}.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved checkpoint: {path}")

    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    train(
        num_updates=args.updates,
        episodes_per_update=args.episodes,
        save_every=args.save_every,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
```

- [ ] **Step 2: Run a smoke test (5 updates, 4 episodes)**

```bash
python -m super_tictactoe.train --updates 5 --episodes 4 --save-every 5
```
Expected: Prints loss values, saves `checkpoints/model_0005.pt` and `checkpoints/model_final.pt`

- [ ] **Step 3: Commit**

```bash
git add super_tictactoe/train.py
git commit -m "feat: training script with checkpointing"
```

---

### Task 10: Evaluation

**Files:**
- Create: `super_tictactoe/evaluate.py`

- [ ] **Step 1: Write the evaluation script**

```python
# super_tictactoe/evaluate.py
import torch
import numpy as np
from typing import Dict
from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.model import ActorCritic


def evaluate(
    model1: ActorCritic,
    model2: ActorCritic,
    num_games: int = 100,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Play num_games between model1 (P1) and model2 (P2).
    Returns win rates: {'model1_wins': float, 'model2_wins': float, 'draws': float}
    """
    env = SuperTicTacToeEnv()
    counts = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0}

    for _ in range(num_games):
        state = env.reset()
        while not env.done:
            model = model1 if env.current_player == 1 else model2
            action_mask = torch.BoolTensor(env.get_action_mask())
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, action_mask, deterministic=True)
            state, _, _, _ = env.step(action)

        if env.winner == 1:
            counts['model1_wins'] += 1
        elif env.winner == 2:
            counts['model2_wins'] += 1
        else:
            counts['draws'] += 1

    return {k: v / num_games for k, v in counts.items()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--games', type=int, default=100)
    args = parser.parse_args()

    m1 = ActorCritic()
    m1.load_state_dict(torch.load(args.model1, map_location='cpu'))
    m2 = ActorCritic()
    m2.load_state_dict(torch.load(args.model2, map_location='cpu'))

    results = evaluate(m1, m2, num_games=args.games)
    print(f"Model1 wins: {results['model1_wins']:.1%}")
    print(f"Model2 wins: {results['model2_wins']:.1%}")
    print(f"Draws:       {results['draws']:.1%}")
```

- [ ] **Step 2: Smoke test against two random models**

```bash
python -m super_tictactoe.train --updates 2 --episodes 2 --save-every 1
python -m super_tictactoe.evaluate --model1 checkpoints/model_0001.pt --model2 checkpoints/model_0002.pt --games 10
```
Expected: Prints win rate percentages without error.

- [ ] **Step 3: Commit**

```bash
git add super_tictactoe/evaluate.py
git commit -m "feat: agent vs agent evaluation with win rate reporting"
```

---

### Task 11: Pygame GUI

**Files:**
- Create: `super_tictactoe/gui.py`

No unit tests — visual component. Manual verification required.

- [ ] **Step 1: Write the GUI**

```python
# super_tictactoe/gui.py
import sys
import time
import pygame
import torch
import numpy as np
from super_tictactoe.env import SuperTicTacToeEnv, GRID_POSITIONS
from super_tictactoe.model import ActorCritic

# ── Layout constants ────────────────────────────────────────────────────────
CELL_SIZE   = 50
CELL_PAD    = 3
MARGIN      = 40
LEVEL_GAP   = 25          # extra vertical space between levels
INFO_HEIGHT = 80          # bottom status bar

COLORS = {
    'bg':        (30,  30,  40),
    'cell':      (60,  60,  80),
    'hover':     (90,  90, 120),
    'p1':        (80, 140, 220),   # blue X
    'p2':        (220, 80,  80),   # red O
    'win_line':  (255, 220,  50),
    'text':      (220, 220, 220),
    'pad':       (30,  30,  40),   # padding cells = invisible
}


def cell_pixel(row: int, col: int):
    """Top-left pixel of cell (row, col) in 12×12 grid."""
    x = MARGIN + col * (CELL_SIZE + CELL_PAD)
    level_offsets = [0, 4 * (CELL_SIZE + CELL_PAD) + LEVEL_GAP,
                     8 * (CELL_SIZE + CELL_PAD) + 2 * LEVEL_GAP]
    if row < 4:
        y = MARGIN + row * (CELL_SIZE + CELL_PAD) + level_offsets[0]
    elif row < 8:
        y = MARGIN + (row - 4) * (CELL_SIZE + CELL_PAD) + level_offsets[1]
    else:
        y = MARGIN + (row - 8) * (CELL_SIZE + CELL_PAD) + level_offsets[2]
    return x, y


def window_size():
    w = MARGIN * 2 + 12 * (CELL_SIZE + CELL_PAD)
    h = MARGIN * 2 + 12 * (CELL_SIZE + CELL_PAD) + 2 * LEVEL_GAP + INFO_HEIGHT
    return w, h


def draw_board(screen, env, font, hover_cell=None, last_placed=None, message=""):
    screen.fill(COLORS['bg'])

    for r in range(12):
        for c in range(12):
            if not env.valid_mask[r, c]:
                continue
            x, y = cell_pixel(r, c)
            color = COLORS['hover'] if (r, c) == hover_cell else COLORS['cell']
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE), border_radius=4)

            piece = env.board[r, c]
            cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
            if piece == 1:
                # Draw X
                offset = CELL_SIZE // 3
                pygame.draw.line(screen, COLORS['p1'], (cx-offset, cy-offset), (cx+offset, cy+offset), 3)
                pygame.draw.line(screen, COLORS['p1'], (cx+offset, cy-offset), (cx-offset, cy+offset), 3)
            elif piece == 2:
                # Draw O
                pygame.draw.circle(screen, COLORS['p2'], (cx, cy), CELL_SIZE // 3, 3)

            # Highlight last placed cell
            if last_placed == (r, c):
                pygame.draw.rect(screen, COLORS['win_line'], (x, y, CELL_SIZE, CELL_SIZE), 3, border_radius=4)

    # Status bar
    status = message or (
        f"Player {'1 (X)' if env.current_player == 1 else '2 (O)'}'s turn"
        if not env.done else
        (f"Player {env.winner} wins!" if env.winner else "Draw!")
    )
    text_surf = font.render(status, True, COLORS['text'])
    _, h = window_size()
    screen.blit(text_surf, (MARGIN, h - INFO_HEIGHT + 10))
    pygame.display.flip()


def get_cell_from_mouse(env, mx, my):
    for r in range(12):
        for c in range(12):
            if not env.valid_mask[r, c]:
                continue
            x, y = cell_pixel(r, c)
            if x <= mx < x + CELL_SIZE and y <= my < y + CELL_SIZE:
                return r, c
    return None


def run_human_vs_agent(model_path: str, human_player: int = 1, device: str = 'cpu'):
    model = ActorCritic().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pygame.init()
    screen = pygame.display.set_mode(window_size())
    pygame.display.set_caption("Super Tic-Tac-Toe — Human vs Agent")
    font = pygame.font.SysFont('monospace', 18)
    clock = pygame.time.Clock()

    env = SuperTicTacToeEnv()
    state = env.reset()
    hover_cell = None
    last_placed = None
    message = ""

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.MOUSEMOTION:
                cell = get_cell_from_mouse(env, *event.pos)
                hover_cell = cell if (cell and env.board[cell[0], cell[1]] == 0) else None

            if event.type == pygame.MOUSEBUTTONDOWN and not env.done:
                if env.current_player == human_player:
                    cell = get_cell_from_mouse(env, *event.pos)
                    if cell and env.get_action_mask()[cell[0] * 12 + cell[1]]:
                        action = cell[0] * 12 + cell[1]
                        state, _, _, info = env.step(action)
                        last_placed = info['placed']
                        message = "Move forfeited!" if info['forfeited'] else ""

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                state = env.reset()
                last_placed = None
                message = ""

        # Agent's turn
        if not env.done and env.current_player != human_player:
            time.sleep(0.4)
            action_mask = torch.BoolTensor(env.get_action_mask())
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, action_mask)
            state, _, _, info = env.step(action)
            last_placed = info['placed']

        draw_board(screen, env, font, hover_cell, last_placed, message)
        clock.tick(30)


def run_agent_vs_agent(model1_path: str, model2_path: str, delay: float = 0.5, device: str = 'cpu'):
    def load(path):
        m = ActorCritic().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        return m

    models = {1: load(model1_path), 2: load(model2_path)}

    pygame.init()
    screen = pygame.display.set_mode(window_size())
    pygame.display.set_caption("Super Tic-Tac-Toe — Agent vs Agent")
    font = pygame.font.SysFont('monospace', 18)
    clock = pygame.time.Clock()

    env = SuperTicTacToeEnv()
    state = env.reset()
    last_placed = None
    last_move_time = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                state = env.reset()
                last_placed = None
                last_move_time = time.time()

        if not env.done and time.time() - last_move_time >= delay:
            model = models[env.current_player]
            action_mask = torch.BoolTensor(env.get_action_mask())
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, action_mask)
            state, _, _, info = env.step(action)
            last_placed = info['placed']
            last_move_time = time.time()

        draw_board(screen, env, font, last_placed=last_placed)
        clock.tick(30)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['human', 'agent'])
    parser.add_argument('--model1', type=str, default='checkpoints/model_final.pt')
    parser.add_argument('--model2', type=str, default='checkpoints/model_final.pt')
    parser.add_argument('--human-player', type=int, default=1, choices=[1, 2])
    parser.add_argument('--delay', type=float, default=0.5)
    args = parser.parse_args()

    if args.mode == 'human':
        run_human_vs_agent(args.model1, args.human_player)
    else:
        run_agent_vs_agent(args.model1, args.model2, args.delay)
```

- [ ] **Step 2: Manually verify GUI launches**

```bash
# Train briefly first if no checkpoint exists
python -m super_tictactoe.train --updates 5 --episodes 4

# Human vs agent (press R to restart)
python -m super_tictactoe.gui human --model1 checkpoints/model_final.pt

# Agent vs agent
python -m super_tictactoe.gui agent --model1 checkpoints/model_final.pt --model2 checkpoints/model_final.pt
```

Verify:
- [ ] Triangle board renders centered at all 3 levels
- [ ] Clicking valid cells works in human mode
- [ ] Agent responds after human move
- [ ] "Move forfeited!" message appears occasionally (stochastic)
- [ ] X/O placed correctly, last placed cell highlighted
- [ ] Win/draw message shown when game ends
- [ ] R key resets the game

- [ ] **Step 3: Commit**

```bash
git add super_tictactoe/gui.py
git commit -m "feat: pygame GUI with human vs agent and agent vs agent modes"
```

---

## Running the Full Pipeline

```bash
# 1. Train
python -m super_tictactoe.train --updates 1000 --episodes 512

# 2. Evaluate two checkpoints against each other
python -m super_tictactoe.evaluate \
  --model1 checkpoints/model_0500.pt \
  --model2 checkpoints/model_final.pt \
  --games 100

# 3. Play against the agent
python -m super_tictactoe.gui human --model1 checkpoints/model_final.pt

# 4. Watch two agents play
python -m super_tictactoe.gui agent \
  --model1 checkpoints/model_final.pt \
  --model2 checkpoints/model_0500.pt
```
