import unittest.mock
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

def test_reset_clears_done_and_winner():
    env = SuperTicTacToeEnv()
    env.done = True
    env.winner = 1
    env.reset()
    assert env.done is False
    assert env.winner is None

def test_state_dtype_float32():
    env = SuperTicTacToeEnv()
    state = env.reset()
    assert state.dtype == np.float32

def test_state_channels_reflect_pieces():
    env = SuperTicTacToeEnv()
    env.reset()
    env.board[0, 4] = 1  # P1 piece at (0,4)
    env.board[0, 5] = 2  # P2 piece at (0,5)
    env.current_player = 1
    state = env._get_state()
    assert state[0, 0, 4] == 1.0  # channel 0 = my (P1) pieces
    assert state[1, 0, 5] == 1.0  # channel 1 = opponent (P2) pieces
    assert state[2, 0, 6] == 1.0  # channel 2 = empty valid cell

def test_state_perspective_flip_for_player2():
    env = SuperTicTacToeEnv()
    env.reset()
    env.board[0, 4] = 1  # P1 piece
    env.board[0, 5] = 2  # P2 piece
    env.current_player = 2
    state = env._get_state()
    assert state[0, 0, 5] == 1.0  # channel 0 = my (P2) pieces
    assert state[1, 0, 4] == 1.0  # channel 1 = opponent (P1) pieces

def test_action_mask_dtype_bool():
    env = SuperTicTacToeEnv()
    env.reset()
    mask = env.get_action_mask()
    assert mask.dtype == bool

def test_action_mask_reflects_occupied_cells():
    env = SuperTicTacToeEnv()
    env.reset()
    env.board[0, 4] = 1  # occupy cell at (0,4) → action 0*12+4 = 4
    mask = env.get_action_mask()
    assert not mask[4]   # action 4 should be invalid
    assert mask.sum() == 95  # one less than 96

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
    """All valid neighbors occupied → only (0,4) or forfeit possible."""
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
    cells = [(3,4),(4,5),(5,6),(6,7),(7,8)]
    assert all(env.valid_mask[r,c] for r,c in cells), "Test cells not all valid"
    _place(env, 1, cells)
    assert env._check_win(1)

def test_diagonal_win_5_topright_to_bottomleft():
    env = SuperTicTacToeEnv()
    env.reset()
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


def test_step_places_piece_on_board():
    env = SuperTicTacToeEnv()
    env.reset()
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
    env.board[0, 4] = 1; env.board[0, 5] = 1; env.board[0, 6] = 1
    env.current_player = 1
    action = 0 * 12 + 7  # (0, 7) → completes 4-in-row in G0
    with unittest.mock.patch('numpy.random.random', return_value=0.1):
        state, reward, done, info = env.step(action)
    assert reward >= 1.0
    assert done
    assert env.winner == 1

def test_step_forfeit_no_piece_placed():
    env = SuperTicTacToeEnv()
    env.reset()
    # Force forfeit: 50% branch at (0,4), direction (-1,-1) goes out of G0
    with unittest.mock.patch('numpy.random.random', return_value=0.6), \
         unittest.mock.patch('numpy.random.randint', return_value=0):  # direction (-1,-1)
        state, reward, done, info = env.step(0 * 12 + 4)
    assert info['forfeited']
    assert env.board[0, 4] == 0  # nothing placed

def test_step_draw_when_board_full():
    env = SuperTicTacToeEnv()
    env.reset()
    cells = list(zip(*np.where(env.valid_mask)))
    for i, (r, c) in enumerate(cells[:-1]):
        env.board[r, c] = 1 if i % 2 == 0 else 2
    env.current_player = 1
    last_r, last_c = cells[-1]
    action = last_r * 12 + last_c
    # Patch _check_win to isolate draw-detection from win-detection;
    # alternating fill incidentally creates winning lines.
    with unittest.mock.patch('numpy.random.random', return_value=0.1), \
         unittest.mock.patch.object(env, '_check_win', return_value=False):
        _, reward, done, _ = env.step(action)
    assert done
    assert env.winner is None


def test_evaluate_board_empty():
    env = SuperTicTacToeEnv()
    env.reset()
    assert env._evaluate_board(1) == 0.0
    assert env._evaluate_board(2) == 0.0

def test_evaluate_board_increases_with_line():
    env = SuperTicTacToeEnv()
    env.reset()
    env.board[0, 4] = 1
    v1 = env._evaluate_board(1)
    env.board[0, 5] = 1
    v2 = env._evaluate_board(1)
    env.board[0, 6] = 1
    v3 = env._evaluate_board(1)
    assert v1 < v2 < v3

def test_evaluate_board_blocked_line_scores_lower():
    env = SuperTicTacToeEnv()
    env.reset()
    env.board[0, 4] = 1
    env.board[0, 5] = 1
    env.board[0, 6] = 1
    v_unblocked = env._evaluate_board(1)
    env.board[0, 7] = 2  # opponent blocks the only 4-cell window
    v_blocked = env._evaluate_board(1)
    assert v_unblocked > v_blocked

def test_step_shaping_positive_for_line_building():
    """Building a line gives a positive reward even without a win."""
    env = SuperTicTacToeEnv()
    env.reset()
    env.board[0, 4] = 1
    env.board[0, 5] = 1
    env.current_player = 1
    action = 0 * 12 + 6  # extends to 3-in-a-row
    with unittest.mock.patch('numpy.random.random', return_value=0.1):
        _, reward, done, _ = env.step(action)
    assert reward > 0
    assert not done

def test_step_shaping_gamma_zero_gives_sparse_reward():
    """With shaping_gamma=0, step() reverts to sparse rewards."""
    env = SuperTicTacToeEnv(shaping_gamma=0.0)
    env.reset()
    env.board[0, 4] = 1
    env.board[0, 5] = 1
    env.current_player = 1
    action = 0 * 12 + 6  # no win
    with unittest.mock.patch('numpy.random.random', return_value=0.1):
        _, reward, _, _ = env.step(action)
    assert reward == 0.0  # no shaping, no win → pure 0
