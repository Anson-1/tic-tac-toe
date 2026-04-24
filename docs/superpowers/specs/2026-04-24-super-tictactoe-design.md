# Super Tic-Tac-Toe: Design Spec

**Date:** 2026-04-24
**Stack:** Python, PyTorch, Pygame
**RL Algorithm:** PPO (Proximal Policy Optimization) with Self-Play

---

## 1. Game Rules

### Board
- 6 sub-grids arranged in a centered triangle:
  ```
  Level 1:  [pad×4] [G0: 4 cols] [pad×4]               (1 grid, centered)
  Level 2:  [pad×2] [G1: 4 cols] [G2: 4 cols] [pad×2]  (2 grids, centered)
  Level 3:  [G3: 4 cols] [G4: 4 cols] [G5: 4 cols]     (3 grids, full width)
  ```
- Each sub-grid is 4×4 = 16 cells
- Total valid cells: 6 × 16 = **96 cells**
- Full grid representation: **12×12** (with padding)

### Win Conditions
- **4 in a row** — 4 consecutive pieces horizontally within any single sub-grid
- **4 in a column** — 4 consecutive pieces in the same column index of the 12×12 grid, where the pieces span at least 2 different levels (rows 0-3 = Level 1, rows 4-7 = Level 2, rows 8-11 = Level 3). Only columns 4-7 span all 3 levels; columns 2-3 and 8-9 span Levels 2-3 only.
- **5 across a diagonal** — 5 consecutive pieces diagonally (top-left→bottom-right or top-right→bottom-left) in the 12×12 grid, crossing sub-grid boundaries

### Stochastic Placement
- **50%** chance: piece placed at chosen cell
- **50%** chance: uniformly pick 1 of 8 adjacent cells (same sub-grid)
  - If selected cell is occupied or out of bounds → move **forfeited**
- Corner example: P(forfeiture) = 1/2 × 5/8 = **5/16**

### Players
- Player 1: X (blue), Player 2: O (red)
- Two players alternate turns
- Draw if board is full with no winner

---

## 2. State Representation

- Shape: `(3, 12, 12)` — 3 channels over the 12×12 spatial grid
  - Channel 0: current player's pieces (1 where placed, 0 elsewhere)
  - Channel 1: opponent's pieces
  - Channel 2: empty valid cells
- Padded cells: always 0 across all channels, permanently masked as invalid
- **Perspective flipping:** agent always sees itself as Player 1 — when it's Player 2's turn, swap channels 0 and 1

---

## 3. Neural Network (Actor-Critic)

```
Input: (3, 12, 12)
    → Conv2d(3→32, kernel=3, padding=1) + ReLU
    → Conv2d(32→64, kernel=3, padding=1) + ReLU
    → Flatten → Linear(9216, 256) + ReLU   # 64 × 12 × 12 = 9216
         ↙                        ↘
Actor head                   Critic head
Linear(256→144)              Linear(256→1)
mask padded + occupied       → V(s): expected future reward
→ Softmax → π(a|s)
```

### Actor
- Outputs probability distribution over 144 actions (one per cell in 12×12 grid)
- Invalid action masking: set logits of padded and occupied cells to `-inf` before softmax → probability exactly 0

### Critic
- Outputs single scalar V(s): estimated expected future reward from current state
- Used only during training to compute advantages

---

## 4. PPO Training Loop

### Self-Play Episode Collection
```
reset board
while not done:
    flip channels if Player 2's turn
    agent picks action from π(a|s)
    apply stochastic placement
    store (state, action, log_prob, value, reward, done)
    switch player
```

### Reward
- Win: `+1`, Loss: `-1`, Draw / ongoing / forfeited move: `0`

### Advantage Estimation (GAE)
- Compute actual returns backwards from end of episode:
  `G_t = r_t + γ × G_{t+1}`
- Advantage: `A_t = G_t - V(s_t)`

### PPO Update (after N episodes)
1. Compute advantages using GAE
2. For K epochs:
   - **Actor loss:** PPO clip (ε=0.2) — prevents too-large policy updates
   - **Critic loss:** MSE between V(s) and actual returns
   - **Entropy bonus:** encourages exploration
3. Clear buffer, collect new episodes

### Hyperparameters
| Parameter           | Value |
|---------------------|-------|
| Episodes per update | 512   |
| PPO epochs          | 4     |
| Clip epsilon        | 0.2   |
| Learning rate       | 3e-4  |
| Entropy coefficient | 0.01  |
| Discount factor γ   | 0.99  |

---

## 5. Pygame GUI

### Two Modes
- **Human vs Agent:** human clicks a cell, agent responds automatically
- **Agent vs Agent:** watch two agents play, configurable speed

### Visual Features
- Centered triangle board layout with correct grid alignment
- Player 1 = X (blue), Player 2 = O (red)
- Hover highlight on valid cells
- Show where piece actually landed (stochastic result)
- "Move forfeited" message when move is forfeited
- Winning line highlighted at end of game

---

## 6. Project Structure

```
super_tictactoe/
├── env.py          ← game environment (board, rules, stochastic placement, win detection)
├── model.py        ← actor-critic CNN
├── ppo.py          ← PPO update logic
├── selfplay.py     ← episode collection with self-play
├── train.py        ← main training script
├── gui.py          ← pygame interface (human vs agent, agent vs agent)
└── evaluate.py     ← win rate tracking, agent vs agent evaluation
```

---

## 7. Bonus

If implemented using TF-Agents, TorchRL, or RLLib, score is multiplied by **1.5 (capped at 50%)**.
Current plan uses PyTorch directly — bonus not targeted.
