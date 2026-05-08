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
    'forfeit':   (255, 140,   0),  # orange — intended cell that was forfeited
    'text':      (220, 220, 220),
}


def cell_pixel(row: int, col: int):
    """Top-left pixel of cell (row, col) in 12×12 grid."""
    x = MARGIN + col * (CELL_SIZE + CELL_PAD)
    if row < 4:
        y = MARGIN + row * (CELL_SIZE + CELL_PAD)
    elif row < 8:
        y = MARGIN + 4 * (CELL_SIZE + CELL_PAD) + LEVEL_GAP + (row - 4) * (CELL_SIZE + CELL_PAD)
    else:
        y = MARGIN + 8 * (CELL_SIZE + CELL_PAD) + 2 * LEVEL_GAP + (row - 8) * (CELL_SIZE + CELL_PAD)
    return x, y


def window_size():
    w = MARGIN * 2 + 12 * (CELL_SIZE + CELL_PAD)
    h = MARGIN * 2 + 12 * (CELL_SIZE + CELL_PAD) + 2 * LEVEL_GAP + INFO_HEIGHT
    return w, h


def draw_board(screen, env, font, hover_cell=None, last_placed=None, forfeit_cell=None, message=""):
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
                offset = CELL_SIZE // 3
                pygame.draw.line(screen, COLORS['p1'], (cx-offset, cy-offset), (cx+offset, cy+offset), 3)
                pygame.draw.line(screen, COLORS['p1'], (cx+offset, cy-offset), (cx-offset, cy+offset), 3)
            elif piece == 2:
                pygame.draw.circle(screen, COLORS['p2'], (cx, cy), CELL_SIZE // 3, 3)

            if last_placed == (r, c):
                pygame.draw.rect(screen, COLORS['win_line'], (x, y, CELL_SIZE, CELL_SIZE), 3, border_radius=4)
            if forfeit_cell == (r, c):
                pygame.draw.rect(screen, COLORS['forfeit'], (x, y, CELL_SIZE, CELL_SIZE), 3, border_radius=4)

    if env.done:
        status = f"Player {env.winner} wins!" if env.winner else "Draw!"
    elif message:
        status = message
    else:
        status = f"Player {'1 (X)' if env.current_player == 1 else '2 (O)'}'s turn"

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


def run_human_vs_agent(model_path: str, human_player: int = 1, device: str = 'cpu',
                       num_simulations: int = 0):
    from super_tictactoe.mcts import MCTS
    model = ActorCritic().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    mcts = MCTS(model, device=device, num_simulations=num_simulations) if num_simulations > 0 else None

    pygame.init()
    screen = pygame.display.set_mode(window_size())
    pygame.display.set_caption("Super Tic-Tac-Toe — Human vs Agent")
    font = pygame.font.SysFont('monospace', 18)
    clock = pygame.time.Clock()

    env = SuperTicTacToeEnv()
    state = env.reset()
    hover_cell = None
    last_placed = None
    forfeit_cell = None
    message = ""
    paused = False  # True while waiting for Space after a forfeit

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = env.reset()
                    last_placed = None
                    forfeit_cell = None
                    message = ""
                    paused = False
                elif event.key == pygame.K_SPACE and paused:
                    paused = False
                    forfeit_cell = None
                    message = ""

            if event.type == pygame.MOUSEMOTION:
                cell = get_cell_from_mouse(env, *event.pos)
                hover_cell = cell if (cell and env.board[cell[0], cell[1]] == 0) else None

            if event.type == pygame.MOUSEBUTTONDOWN and not env.done and not paused:
                if env.current_player == human_player:
                    cell = get_cell_from_mouse(env, *event.pos)
                    if cell and env.get_action_mask()[cell[0] * 12 + cell[1]]:
                        action = cell[0] * 12 + cell[1]
                        state, _, _, info = env.step(action)
                        last_placed = info['placed']
                        if info['forfeited']:
                            forfeit_cell = (action // 12, action % 12)
                            message = "Your move forfeited! (drifted out of bounds)  [Space to continue]"
                            paused = True
                        else:
                            forfeit_cell = None
                            message = ""

        if not env.done and not paused and env.current_player != human_player:
            time.sleep(0.4)
            if mcts:
                action = mcts.get_action(env)
            else:
                action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
                state_tensor = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action, _, _ = model.get_action(state_tensor, action_mask)
            state, _, _, info = env.step(action)
            last_placed = info['placed']
            if info['forfeited']:
                forfeit_cell = (action // 12, action % 12)
                message = f"Agent forfeit! (intended {action // 12},{action % 12})  [Space to continue]"
                paused = True
            else:
                forfeit_cell = None
                message = ""

        draw_board(screen, env, font, hover_cell, last_placed, forfeit_cell, message)
        clock.tick(30)


def run_agent_vs_heuristic(model_path: str, heuristic_name: str = 'blocking',
                            agent_player: int = 1, delay: float = 0.5, device: str = 'cpu'):
    from super_tictactoe.heuristics import greedy_agent, blocking_agent, safe_agent
    import random as _random
    import numpy as _np

    def random_agent(env):
        mask = env.get_action_mask()
        valid = _np.where(mask)[0]
        return int(_random.choice(valid))

    heuristics = {
        'random':   random_agent,
        'greedy':   greedy_agent,
        'blocking': blocking_agent,
        'safe':     safe_agent,
    }
    if heuristic_name not in heuristics:
        print(f"Unknown heuristic '{heuristic_name}'. Choose: {list(heuristics)}")
        return
    heuristic = heuristics[heuristic_name]

    model = ActorCritic().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    heuristic_player = 3 - agent_player

    pygame.init()
    screen = pygame.display.set_mode(window_size())
    pygame.display.set_caption(
        f"Super Tic-Tac-Toe — PPO (P{agent_player}) vs {heuristic_name.capitalize()} (P{heuristic_player})"
    )
    font   = pygame.font.SysFont('monospace', 18)
    sfont  = pygame.font.SysFont('monospace', 14)
    clock  = pygame.time.Clock()

    wins   = {agent_player: 0, heuristic_player: 0, 'draw': 0}
    game_n = 0

    env   = SuperTicTacToeEnv()
    state = env.reset()
    last_placed  = None
    forfeit_cell = None
    paused       = False
    last_move_time = time.time()
    auto_restart = True   # restart automatically after a game ends

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = env.reset()
                    last_placed  = None
                    forfeit_cell = None
                    paused       = False
                    last_move_time = time.time()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    last_move_time = time.time()
                elif event.key == pygame.K_a:
                    auto_restart = not auto_restart

        # Auto-restart when game ends
        if env.done and auto_restart and time.time() - last_move_time >= delay * 2:
            if env.winner:
                wins[env.winner] += 1
            else:
                wins['draw'] += 1
            game_n += 1
            state = env.reset()
            last_placed  = None
            forfeit_cell = None
            last_move_time = time.time()

        if not env.done and not paused and time.time() - last_move_time >= delay:
            who = "PPO" if env.current_player == agent_player else heuristic_name.capitalize()
            intended_r, intended_c = None, None

            if env.current_player == agent_player:
                action_mask  = torch.BoolTensor(env.get_action_mask()).to(device)
                state_tensor = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action, _, _ = model.get_action(state_tensor, action_mask)
            else:
                action = heuristic(env)

            intended_r, intended_c = action // 12, action % 12
            state, _, _, info = env.step(action)
            last_placed = info['placed']

            if info['forfeited']:
                forfeit_cell = (intended_r, intended_c)
                print(f"[Game {game_n+1}] FORFEIT  {who} (P{env.current_player ^ 3}) "
                      f"intended ({intended_r},{intended_c}) → drifted out/occupied, turn wasted")
                paused = True
            else:
                placed_r, placed_c = info['placed']
                drifted = (placed_r != intended_r or placed_c != intended_c)
                drift_str = f"→ drifted to ({placed_r},{placed_c})" if drifted else "→ landed exactly"
                print(f"[Game {game_n+1}] MOVE     {who} (P{env.current_player ^ 3}) "
                      f"intended ({intended_r},{intended_c}) {drift_str}"
                      + (f"  WIN!" if env.done and env.winner else ""))
                forfeit_cell = None

            last_move_time = time.time()

        # Draw
        screen.fill(COLORS['bg'])

        for r in range(12):
            for c in range(12):
                if not env.valid_mask[r, c]:
                    continue
                x, y  = cell_pixel(r, c)
                color = COLORS['cell']
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE), border_radius=4)

                piece = env.board[r, c]
                cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
                if piece == 1:
                    off = CELL_SIZE // 3
                    pygame.draw.line(screen, COLORS['p1'], (cx-off, cy-off), (cx+off, cy+off), 3)
                    pygame.draw.line(screen, COLORS['p1'], (cx+off, cy-off), (cx-off, cy+off), 3)
                elif piece == 2:
                    pygame.draw.circle(screen, COLORS['p2'], (cx, cy), CELL_SIZE // 3, 3)

                if last_placed == (r, c):
                    pygame.draw.rect(screen, COLORS['win_line'], (x, y, CELL_SIZE, CELL_SIZE), 3, border_radius=4)
                if forfeit_cell == (r, c):
                    pygame.draw.rect(screen, COLORS['forfeit'], (x, y, CELL_SIZE, CELL_SIZE), 3, border_radius=4)

        # Status bar
        _, h = window_size()
        y0 = h - INFO_HEIGHT + 6
        if env.done:
            if env.winner == agent_player:
                msg = "PPO wins!"
            elif env.winner == heuristic_player:
                msg = f"{heuristic_name.capitalize()} wins!"
            else:
                msg = "Draw!"
        elif paused and forfeit_cell:
            msg = f"FORFEIT at ({forfeit_cell[0]},{forfeit_cell[1]}) — orange cell — Space to resume"
        elif paused:
            msg = "PAUSED — Space to resume"
        else:
            who = "PPO" if env.current_player == agent_player else heuristic_name.capitalize()
            msg = f"{who}'s turn (P{env.current_player})"

        screen.blit(font.render(msg, True, COLORS['text']), (MARGIN, y0))

        score_str = (f"Game {game_n+1}  |  "
                     f"PPO: {wins[agent_player]}  "
                     f"{heuristic_name.capitalize()}: {wins[heuristic_player]}  "
                     f"Draw: {wins['draw']}  |  "
                     f"[R] restart  [Space] pause  [A] auto={'ON' if auto_restart else 'OFF'}")
        screen.blit(sfont.render(score_str, True, COLORS['text']), (MARGIN, y0 + 22))

        pygame.display.flip()
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
    forfeit_cell = None
    message = ""
    paused = False
    last_move_time = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = env.reset()
                    last_placed = None
                    forfeit_cell = None
                    message = ""
                    paused = False
                    last_move_time = time.time()
                elif event.key == pygame.K_SPACE and paused:
                    paused = False
                    forfeit_cell = None
                    message = ""
                    last_move_time = time.time()

        if not env.done and not paused and time.time() - last_move_time >= delay:
            model = models[env.current_player]
            action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, action_mask)
            state, _, _, info = env.step(action)
            last_placed = info['placed']
            if info['forfeited']:
                forfeit_cell = (action // 12, action % 12)
                message = f"P{3 - env.current_player} forfeit! (intended {action // 12},{action % 12})  [Space to continue]"
                paused = True
            else:
                forfeit_cell = None
                message = ""
                last_move_time = time.time()

        draw_board(screen, env, font, last_placed=last_placed, forfeit_cell=forfeit_cell, message=message)
        clock.tick(30)


def run_human_vs_heuristic(heuristic_name: str = 'horizontal', human_player: int = 1):
    from super_tictactoe.heuristics import (
        greedy_agent, blocking_agent, safe_agent, horizontal_agent,
    )
    import random as _random

    def random_agent(env):
        mask = env.get_action_mask()
        valid = np.where(mask)[0]
        return int(_random.choice(valid))

    heuristics = {
        'random': random_agent,
        'greedy': greedy_agent,
        'blocking': blocking_agent,
        'safe': safe_agent,
        'horizontal': horizontal_agent,
    }
    heuristic = heuristics[heuristic_name]

    pygame.init()
    screen = pygame.display.set_mode(window_size())
    pygame.display.set_caption(f"Super Tic-Tac-Toe — Human vs {heuristic_name.capitalize()}")
    font = pygame.font.SysFont('monospace', 18)
    clock = pygame.time.Clock()

    env = SuperTicTacToeEnv()
    state = env.reset()
    hover_cell = None
    last_placed = None
    forfeit_cell = None
    message = ""
    paused = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = env.reset()
                    last_placed = None
                    forfeit_cell = None
                    message = ""
                    paused = False
                elif event.key == pygame.K_SPACE and paused:
                    paused = False
                    forfeit_cell = None
                    message = ""

            if event.type == pygame.MOUSEMOTION:
                cell = get_cell_from_mouse(env, *event.pos)
                hover_cell = cell if (cell and env.board[cell[0], cell[1]] == 0) else None

            if event.type == pygame.MOUSEBUTTONDOWN and not env.done and not paused:
                if env.current_player == human_player:
                    cell = get_cell_from_mouse(env, *event.pos)
                    if cell and env.get_action_mask()[cell[0] * 12 + cell[1]]:
                        action = cell[0] * 12 + cell[1]
                        state, _, _, info = env.step(action)
                        last_placed = info['placed']
                        if info['forfeited']:
                            forfeit_cell = (action // 12, action % 12)
                            message = "Your move forfeited!  [Space to continue]"
                            paused = True
                        else:
                            forfeit_cell = None
                            message = ""

        if not env.done and not paused and env.current_player != human_player:
            time.sleep(0.4)
            action = heuristic(env)
            state, _, _, info = env.step(action)
            last_placed = info['placed']
            if info['forfeited']:
                forfeit_cell = (action // 12, action % 12)
                message = f"{heuristic_name.capitalize()} forfeit!  [Space to continue]"
                paused = True
            else:
                forfeit_cell = None
                message = ""

        draw_board(screen, env, font, hover_cell, last_placed, forfeit_cell, message)
        clock.tick(30)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['human', 'agent', 'heuristic', 'human_vs_heuristic'])
    parser.add_argument('--model1', type=str, default='checkpoints/model_final.pt')
    parser.add_argument('--model2', type=str, default='checkpoints/model_final.pt')
    parser.add_argument('--human-player', type=int, default=1, choices=[1, 2])
    parser.add_argument('--agent-player', type=int, default=1, choices=[1, 2],
                        help='Which player slot the PPO agent occupies (heuristic mode)')
    parser.add_argument('--heuristic', type=str, default='blocking',
                        choices=['random', 'greedy', 'blocking', 'safe', 'horizontal'],
                        help='Heuristic opponent')
    parser.add_argument('--delay', type=float, default=0.5)
    parser.add_argument('--simulations', type=int, default=0,
                        help='MCTS simulations per move (0 = direct policy, no MCTS)')
    args = parser.parse_args()

    if args.mode == 'human':
        run_human_vs_agent(args.model1, args.human_player, num_simulations=args.simulations)
    elif args.mode == 'heuristic':
        run_agent_vs_heuristic(args.model1, args.heuristic, args.agent_player, args.delay)
    elif args.mode == 'human_vs_heuristic':
        run_human_vs_heuristic(args.heuristic, args.human_player)
    else:
        run_agent_vs_agent(args.model1, args.model2, args.delay)
