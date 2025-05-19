import bases
import logging
import csv
from .agent import QLearningAgent
from .player import HumanPlayer
from typing import Literal


class Bases:
    game_counter = 0

    def __init__(self, size):
        self.size = size
        self.lines = []
        self.boxes = [[None] * size for _ in range(size)]
        self.scores = {'A': 0, 'B': 0}
        self.wins = {'A': 0, 'B': 0, 'Tie': 0}
        self.turn = 'A'
        self.turn_id = 1
        self.game_id = None
        logging.basicConfig(
            filename=bases.cfg['game_log'],
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.info(f'Initialized Bases game with size {size}x{size}')

    def reset(self) -> None:
        self.lines = []
        self.boxes = [[None] * self.size for _ in range(self.size)]
        self.scores = {'A': 0, 'B': 0}
        self.turn = 'A'
        self.turn_id = 1

    def print_board(self) -> None:
        delimiter = "-" * (self.size * 4 + 1)
        board = delimiter + "\n"
        for row in range(self.size):
            for col in range(self.size):
                board += '*'
                if self.is_line_drawn(col, row, col + 1, row):
                    board += '---'
                else:
                    board += '   '
            board += '*\n'
            for col in range(self.size + 1):
                if self.is_line_drawn(col, row, col, row + 1):
                    board += '|'
                else:
                    board += ' '
                if col < self.size:
                    box = self.boxes[row][col]
                    board += f' {box if box else " "} '
            board += '\n'
        for col in range(self.size):
            board += '*'
            if self.is_line_drawn(col, self.size, col + 1, self.size):
                board += '---'
            else:
                board += '   '
        board += '*\n' + delimiter
        logging.debug("Board state:\n" + board)
        print(board)

    def is_line_drawn(self, x1, y1, x2, y2) -> bool:
        """x, y are column, row"""
        return any(
            (line['x1'], line['y1'], line['x2'], line['y2']) in
            [(x1, y1, x2, y2), (x2, y2, x1, y1)]
            for line in self.lines
        )

    def check_and_update_box(self, row, col) -> bool:
        if (self.is_line_drawn(col, row, col + 1, row) and
                self.is_line_drawn(col, row+1, col + 1, row+1) and
                self.is_line_drawn(col, row, col, row + 1) and
                self.is_line_drawn(col + 1, row, col + 1, row + 1) and
                self.boxes[row][col] is None):
            self.boxes[row][col] = self.turn
            self.scores[self.turn] += 1
            return True
        return False

    def available_moves(self) -> list:
        moves = []
        # Horizontal lines
        for row in range(self.size + 1):
            for col in range(self.size):
                if not self.is_line_drawn(col, row, col+1, row):
                    moves.append((col, row, col+1, row))
        # Vertical lines
        for col in range(self.size + 1):
            for row in range(self.size):
                if not self.is_line_drawn(col, row, col, row+1):
                    moves.append((col, row, col, row+1))
        return moves

    def make_move(self, x1, y1, x2, y2) -> tuple[bool, bool, int]:
        """x, y are column, row."""
        if x1 == x2 and y1 == y2:
            logging.warning("Invalid move attempt with identical coordinates")
            print("Invalid move; coordinates are identical.")
            return False, False, 0
        if (abs(x1 - x2) == 1 and y1 == y2) or \
                (x1 == x2 and abs(y1 - y2) == 1):
            if self.is_line_drawn(x1, y1, x2, y2):
                logging.warning("Invalid move attempt on existing line")
                print("Invalid move; line already exists.")
                return False, False, 0
            self.lines.append({
                'game_id': self.game_id,
                'turn_id': self.turn_id,
                'player': self.turn,
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            })
            self.turn_id += 1
            logging.info(f"Player {self.turn} drew a line from \
                         ({x1}, {y1}) to ({x2}, {y2})")
            completed_box = any(
                self.check_and_update_box(row, col)
                for row in range(self.size)
                for col in range(self.size)
            )
            return True, completed_box, self.is_potential_box(x1, y1, x2, y2)
        else:
            logging.warning("Invalid move attempt with non-adjacent points")
            print("Invalid move: points must be adjacent",
                  "and form a straight line.")
            return False, False, 0

    def get_state(self):
        return tuple(sorted(
            ((line['x1'], line['y1'],
              line['x2'], line['y2']) for line in self.lines)
        ))

    def is_full(self) -> bool:
        return len(self.lines) == 2 * self.size * (self.size + 1)

    def save_lines_to_csv(self) -> None:
        file_exists = False
        try:
            with open(bases.LINES_LOG_PATH, 'r'):
                file_exists = True
        except FileNotFoundError:
            pass
        with open(bases.LINES_LOG_PATH, 'a', newline='') as csvfile:
            fieldnames = ['game_id',
                          'turn_id',
                          'player',
                          'x1', 'y1', 'x2', 'y2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for line in self.lines:
                writer.writerow(line)
        logging.info("Lines saved to %s", bases.LINES_LOG_PATH)

    def print_win_counts(self) -> None:
        print(f"Games won by A: {self.wins['A']}, B: {self.wins['B']}",
              f"Ties: {self.wins['Tie']}")

    def play(self, player_a, player_b) -> Literal['A', 'B', 'Tie']:
        """Run a game between two players (agents or human)."""
        self.reset()
        Bases.game_counter += 1
        self.game_id = Bases.game_counter
        logging.info(f'Starting game {self.game_id}')
        print("\nStarting new game...")

        self.print_win_counts()
        last_state = None
        last_action = None

        try:
            while not self.is_full():
                self.print_board()
                # Determine the move: AI/Human
                if self.turn == 'A':
                    move = player_a.choose_action(self)
                    if isinstance(player_a, QLearningAgent):
                        print(f"AI (Player A)'s turn: drawing line {move}")
                else:
                    move = player_b.choose_action(self)
                    if isinstance(player_b, QLearningAgent):
                        print(f"AI (Player B)'s turn: drawing line {move}")

                # Record state/action for learning agent
                state = self.get_detailed_state()
                if isinstance(player_a, QLearningAgent) and self.turn == 'A':
                    last_state = state
                    last_action = move
                elif isinstance(player_b, QLearningAgent) and self.turn == 'B':
                    last_state = state
                    last_action = move

                # Try move
                success, completed_box, potential_box = self.make_move(*move)
                if not success:
                    continue

                # Reward logic for Q-learning agents
                reward = 0
                if completed_box:
                    reward = 1

                next_state = self.get_detailed_state()
                available_moves = self.available_moves()

                if isinstance(player_a, QLearningAgent) and self.turn == 'A':
                    player_a.update(last_state, last_action,
                                    reward, next_state, available_moves)
                elif isinstance(player_b, QLearningAgent) and self.turn == 'B':
                    player_b.update(last_state, last_action,
                                    reward, next_state, available_moves)

                # Alternate turns regardless of box completion
                self.turn = 'B' if self.turn == 'A' else 'A'

        except KeyboardInterrupt:
            print("\nGame interrupted.")
            exit()
        finally:
            self.print_board()
            self.save_lines_to_csv()
            print("Game progress saved to lines.csv.")

        # Determine winner
        if self.scores['A'] > self.scores['B']:
            result = 'A'
        elif self.scores['A'] < self.scores['B']:
            result = 'B'
        else:
            result = 'Tie'

        self.wins[result] += 1
        print(f"Game over. Result: {result}")
        self.print_win_counts()
        return result

    def game_loop(self) -> None:
        """Game mode selection."""
        print("Select game mode:")
        print("1. Human vs Human")
        print("2. Human vs AI (Human as Player A)")
        print("3. AI vs Human (Human as Player B)")
        choice = int(input("Enter the number corresponding to your choice: ")
                     .strip())

        if choice == 1:
            while True:
                result = self.play(HumanPlayer(), HumanPlayer())
                print(f"Game over. Result: {result}")
        elif choice == 2:
            agent = QLearningAgent(self.size, training_mode=False)
            agent.load_policy(bases.POLICY_PATH)
            while True:
                result = self.play(HumanPlayer(), agent)  # human_player='A'
                print(f"Game over. Result: {result}")
        elif choice == 3:
            agent = QLearningAgent(self.size, training_mode=False)
            agent.load_policy(bases.POLICY_PATH)
            while True:
                result = self.play(agent, HumanPlayer())  # human_player='B'
                print(f"Game over. Result: {result}")
        else:
            print("Invalid choice, exiting game loop.")

    def is_potential_box(self, x1, y1, x2, y2) -> int:
        potential_boxes = 0
        # Is it a vertical line? (col stays constant, row changes)
        if x1 == x2:
            col = x1
            for row in [y1 - 1, y1]:
                if 0 <= row < self.size:
                    if (
                        self.is_line_drawn(col, row, col + 1, row)
                        and self.is_line_drawn(col, row + 1, col + 1, row + 1)
                        and self.is_line_drawn(col, row, col, row + 1)
                        and self.is_line_drawn(col + 1, row, col + 1, row + 1)
                    ):
                        potential_boxes += 1

        # Is it a horizontal line? (row stays constant, col changes)
        if y1 == y2:
            row = y1
            for col in [x1 - 1, x1]:
                if 0 <= col < self.size:
                    if (
                        self.is_line_drawn(col, row, col + 1, row)
                        and self.is_line_drawn(col, row + 1, col + 1, row + 1)
                        and self.is_line_drawn(col, row, col, row + 1)
                        and self.is_line_drawn(col + 1, row, col + 1, row + 1)
                    ):
                        potential_boxes += 1
        return potential_boxes

    def get_detailed_state(self):
        """
        State is:
          - List of drawn lines ((x1,y1,x2,y2) tuples)
          - Ownership of boxes [[row][col]]
        """
        return (
            tuple(sorted(((line['x1'], line['y1'], line['x2'], line['y2'])
                          for line in self.lines))),
            tuple(tuple(self.boxes[row][col] for col in range(self.size))
                  for row in range(self.size))
        )
