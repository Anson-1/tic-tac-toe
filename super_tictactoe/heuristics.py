import random
import numpy as np
from super_tictactoe.env import SuperTicTacToeEnv


def greedy_agent(env):
    """Pure offensive: win immediately if possible, else maximise own potential. No blocking."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    me = env.current_player

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    best_a, best_v = valid[0], -1.0
    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        v = env._evaluate_board(me)
        env.board[r, c] = 0
        if v > best_v:
            best_v, best_a = v, a
    return int(best_a)


def blocking_agent(env):
    """Win if possible, block opponent's immediate win, else maximise own potential."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    me  = env.current_player
    opp = 3 - me

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = opp
        if env._check_win(opp):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    best_a, best_v = valid[0], -1.0
    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        v = env._evaluate_board(me)
        env.board[r, c] = 0
        if v > best_v:
            best_v, best_a = v, a
    return int(best_a)


def safe_agent(env):
    """Win/block first, then prefer cells with more valid neighbours (lower forfeit risk)."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    me  = env.current_player
    opp = 3 - me

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = opp
        if env._check_win(opp):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    def valid_neighbour_count(a):
        r, c = a // 12, a % 12
        return sum(
            1 for dr, dc in SuperTicTacToeEnv._DIRECTIONS
            if 0 <= r+dr < 12 and 0 <= c+dc < 12
            and env.valid_mask[r+dr, c+dc]
        )

    best_a, best_v = valid[0], -1.0
    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        v = env._evaluate_board(me) + 0.05 * valid_neighbour_count(a)
        env.board[r, c] = 0
        if v > best_v:
            best_v, best_a = v, a
    return int(best_a)


def one_step_lookahead_agent(env):
    """Depth-2 minimax: win/block immediately, then pick move that maximises
    own potential minus opponent's best greedy response."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    me  = env.current_player
    opp = 3 - me

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = opp
        if env._check_win(opp):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    # Pre-rank by own potential, only search top 20
    scored = []
    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        scored.append((env._evaluate_board(me), a))
        env.board[r, c] = 0
    scored.sort(reverse=True)
    candidates = [a for _, a in scored[:20]]

    best_a, best_score = candidates[0], float('-inf')
    for a in candidates:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        my_val = env._evaluate_board(me)

        rows, cols = np.where(env.valid_mask & (env.board == 0))
        opp_valid = rows * 12 + cols
        opp_best = 0.0
        for oa in opp_valid:
            or_, oc = oa // 12, oa % 12
            env.board[or_, oc] = opp
            v = env._evaluate_board(opp)
            env.board[or_, oc] = 0
            if v > opp_best:
                opp_best = v

        score = my_val - opp_best
        env.board[r, c] = 0
        if score > best_score:
            best_score, best_a = score, a
    return int(best_a)


def counter_heuristic(env):
    """Win immediately, then block opponent's most advanced line (any direction), else greedy."""
    mask = env.get_action_mask()
    valid_set = set(np.where(mask)[0])
    me  = env.current_player
    opp = 3 - me

    # Win immediately
    for a in valid_set:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    # Build all real winning windows (same as _evaluate_board)
    windows = []
    # Horizontal (4)
    for r in range(12):
        for c in range(9):
            cells = [(r, c+i) for i in range(4)]
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                windows.append(cells)
    # Vertical cross-level (4)
    for c in range(12):
        for r in range(9):
            cells = [(r+i, c) for i in range(4)]
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                levels = {env._get_level(rr) for rr, _ in cells}
                if len(levels) >= 2:
                    windows.append(cells)
    # Diagonal ↘ (5)
    for r in range(8):
        for c in range(8):
            cells = [(r+i, c+i) for i in range(5)]
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                windows.append(cells)
    # Diagonal ↙ (5)
    for r in range(8):
        for c in range(4, 12):
            cells = [(r+i, c-i) for i in range(5)]
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                windows.append(cells)

    # Find window with most opponent pieces (unblocked by me)
    best_block, best_count = None, 0
    for cells in windows:
        opp_count = sum(1 for rr, cc in cells if env.board[rr, cc] == opp)
        own_count  = sum(1 for rr, cc in cells if env.board[rr, cc] == me)
        if own_count > 0:
            continue  # already blocked
        empty_cells = [rr * 12 + cc for rr, cc in cells if env.board[rr, cc] == 0
                       and (rr * 12 + cc) in valid_set]
        if opp_count > best_count and empty_cells:
            best_count = opp_count
            # Pick the empty cell in this window closest to the center of the window
            best_block = empty_cells[len(empty_cells) // 2]

    if best_count >= 2:
        return int(best_block)

    return greedy_agent(env)


HEURISTIC_POOL = [
    (greedy_agent,      0.20),
    (blocking_agent,    0.35),
    (safe_agent,        0.30),
    (counter_heuristic, 0.15),
]

_agents  = [h for h, _ in HEURISTIC_POOL]
_weights = [w for _, w in HEURISTIC_POOL]


def random_heuristic(env):
    """Randomly sample a heuristic opponent each call."""
    agent = random.choices(_agents, weights=_weights, k=1)[0]
    return agent(env)
