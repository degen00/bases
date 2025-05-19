# 🎯 Bases

Bases (literal translation from Polish) is currently implemented as an almost classic 'Dots and Boxes'; a pencil-and-paper game played on a grid of dots. Players alternate turns drawing horizontal or vertical lines between adjacent dots. Completing the fourth side of a 1x1 box earns the player ownership of that box. Once the grid is full and no more moves can be made, the player owning the most boxes wins.

## Overview

This game allows you to play, train, and optimize a basic reinforcement learning (Q-learning) or AI driven player. You can:

- Train a Q-learning agent;
- Tune the agent's learning parameters for optimal performance;
- Play the game as a human against another human or the AI.

---
## Release Log
### **v0.1** 
* Initial release (dots and boxes) with text terminal interface


---
---

# 📖 **User Documentation**

## How to Use

### **Quick Setup & Start**

1. Ensure you have Python 3.x installed.
2. Install dependencies: `pip install -r requirements.txt`
3. Start game: `python play.py`

---

### **Modes of Operation**

Upon running, you will be prompted for the grid size and which mode to run e.g.,:

```
Enter the size of the grid: 4
Enter 'train', 'tune', or 'play':
```

*OBS: To play against an AI, first __train__ the Q-policy*

**Modes:**
- `train` : Train the AI agent.
- `tune`  : Tune the hyperparameters (learning rate, etc.) for the AI.
- `play`  : Play the game (against a human opponent or AI agent).

---

#### **Training the AI**
- Choose `train` mode.
- Enter the number of training episodes (games).
- Progress will be saved in `data/policy/policy.json`.

#### **Tuning Hyperparameters**

**OBS** default hyperparameters for a 3x3, 4x4, and 5x5 boards are already available in the config file `bases.yml`

- Choose `tune` mode.
- You can select how many parameter sets to try and how long to train/test each set.
- The best parameters found will be displayed, and all results saved to `data/hp/hyperparameter_results.csv`.

#### **Playing the Game**
- Choose `play` mode.
- Select game mode:
    1. Human vs Human
    2. Human vs AI (you play as Player A)
    3. AI vs Human (you play as Player B)
- If playing with AI, ensure `policy.json` (trained AI policy) exists.

#### **Rules and Input**

- Input moves as four integers (greater of equal to 0) separated by spaces`0 1 0 2` (format: x1 y1 x2 y2). 
  * in the board below: top-left corner is (0, 0) and bottom right corner is (3, 3)
- Moves must draw a straight line between adjacent dots.
- The game ends when the board is full; the player with the most boxes wins.
- The board is printed after each move for visual feedback.


**Example**
```
Player A, enter your move (e.g., '0 0 0 1'): 0 0 1 0
-------------
*---*   *   *
             
*   *   *   *
             
*   *   *   *
             
*   *   *   *
-------------
AI (Player B)'s turn: drawing line (2, 2, 3, 2)
-------------
*---*   *   *
             
*   *   *   *
             
*   *   *---*
             
*   *   *   *
-------------
```
---

### **Logging**

- **policy.json** : Q-learning table for the AI.
- **lines.csv** : Logs moves for each game.
- **hyperparameter_results.csv** : Saves tuning results.
- **game.log** : Detailed log for game events and debugging.

---
---

# ⚙️ **Developer Documentation**

## Architecture Overview

**Main entities:**

- `Bases`: Encapsulates the game logic, state, and interactions.
- `QLearningAgent`: Q-learning reinforcement learning agent.
- `RandomAgent`: Baseline random-action AI.
- `HumanPlayer`: Interactive human input handler.
- `train_agents`, `hyperparameter_tuning`: Utilities for model training and hyperparameter optimization.

---

## Key Methods and Classes

### *Bases(size)*

- Main game controller; manages state (`lines`, `boxes`), move validation, scorekeeping, and game loop.
- Logging via `logging` module to `game.log`.
- Methods:
    - `play(player_a, player_b)`: Executes a single game between two agents/players.
    - `is_line_drawn`, `make_move`, `available_moves`: Manage board state.
    - `save_lines_to_csv`: Persists move history.
    - `print_board`: Renders current board in ASCII to terminal.
    - `get_detailed_state`: Returns a canonical (state, box ownership) tuple.

### *QLearningAgent(grid_size, ...)*

- Holds Q-table as `{(state, move): q_value}` mapping.
- Key methods:
    - `choose_action(game)`: Epsilon-greedy action selection.
    - `update(previous_state, action, reward, next_state, available_moves)`: Standard Q-learning update.
    - `enhance_reward(completed_box, potential_box_opponent)`: Domain-specific reward shaping.
    - `save_policy(path)`, `load_policy(path)`: Persistence.
    - Symmetry-handling: States are transformed to their canonical form via all rotations/reflections to increase learning efficiency.

### *RandomAgent*

- Moves randomly; baseline for tuning/testing.

### *HumanPlayer*

- Prompts user interactively for valid moves.

---

## Hyperparameter Tuning

- `hyperparameter_tuning()` tries random values (within normal ranges) for learning rate, discount, and exploration, then evaluates by win rate against a baseline.
- Results are saved to CSV for later review.

---

## Training

- `train_agents()` runs agent vs. agent games, updating Q-tables.
- The best agent’s policy is saved for use during play.

---

## Adding Features/Modifying Code

- To alter reward shaping, adjust `QLearningAgent.enhance_reward`.
- To extend AI architecture (e.g., use Deep Q-Learning), adapt the `QLearningAgent`.
- To modify game rendering, change `print_board`.

---

## Extending

- Swap out `QLearningAgent` for alternative agents for experimentation.
- Add more sophisticated adversaries (minimax, MCTS, etc.).

---



