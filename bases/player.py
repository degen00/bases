class HumanPlayer:
    def choose_action(self, game) -> tuple[int, int, int, int]:
        """Prompt human player for their move and return it as a tuple."""
        while True:
            try:
                move = input(f"Player {game.turn},"
                             "enter your move (e.g., '0 0 0 1'): ")
                col1, row1, col2, row2 = map(int, move.split())
                if not game.is_line_drawn(col1, row1, col2, row2):
                    return (col1, row1, col2, row2)
                else:
                    print("Invalid move; line already exists. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter four integers",
                      "separated by spaces.")
