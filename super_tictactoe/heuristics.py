import numpy as np


def blocking_agent(env):
    """
    Win if possible, block opponent's immediate win, else maximise own potential.
    Used as a heuristic training opponent to force the model to learn blocking.
    """
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
