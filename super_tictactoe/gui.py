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
                offset = CELL_SIZE // 3
                pygame.draw.line(screen, COLORS['p1'], (cx-offset, cy-offset), (cx+offset, cy+offset), 3)
                pygame.draw.line(screen, COLORS['p1'], (cx+offset, cy-offset), (cx-offset, cy+offset), 3)
            elif piece == 2:
                pygame.draw.circle(screen, COLORS['p2'], (cx, cy), CELL_SIZE // 3, 3)

            if last_placed == (r, c):
                pygame.draw.rect(screen, COLORS['win_line'], (x, y, CELL_SIZE, CELL_SIZE), 3, border_radius=4)

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

        if not env.done and env.current_player != human_player:
            time.sleep(0.4)
            action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
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
            action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
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
