import numpy as np
from typing import Optional, Tuple

# (row_start, col_start) in 12x12 grid for each sub-grid
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

    _DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def _get_grid(self, row: int, col: int) -> Optional[int]:
        for i, (r, c) in enumerate(GRID_POSITIONS):
            if r <= row < r + 4 and c <= col < c + 4:
                return i
        return None

    def _stochastic_place(self, row: int, col: int) -> Tuple[Optional[int], Optional[int]]:
        """
        50%: returns (row, col).
        50%: picks one of 8 adjacent cells uniformly within same sub-grid.
             Returns (None, None) if out of sub-grid bounds or occupied.
        """
        if np.random.random() < 0.5:
            return row, col

        dr, dc = self._DIRECTIONS[np.random.randint(8)]
        new_row, new_col = row + dr, col + dc

        grid_id = self._get_grid(row, col)
        if grid_id is None:
            return None, None
        r_start, c_start = GRID_POSITIONS[grid_id]
        if not (r_start <= new_row < r_start + 4 and c_start <= new_col < c_start + 4):
            return None, None
        if self.board[new_row, new_col] != 0:
            return None, None
        return new_row, new_col

    def _get_level(self, row: int) -> int:
        if row <= 3: return 0
        if row <= 7: return 1
        return 2

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
