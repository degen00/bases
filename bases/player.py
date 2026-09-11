"""Interactive terminal player."""
from __future__ import annotations

from .board import Board


class HumanPlayer:
    name = "human"

    def choose_index(self, board: Board) -> int:
        """Prompt for a move as four integers ``x1 y1 x2 y2`` and validate it."""
        while True:
            raw = input(f"Player {board.turn}, enter your move (e.g. '0 0 1 0'): ")
            parts = raw.split()
            if len(parts) != 4:
                print("Please enter four integers separated by spaces.")
                continue
            try:
                move = tuple(int(p) for p in parts)
                idx = board.geom.index_of(move)
            except ValueError as exc:
                print(f"Invalid move: {exc}")
                continue
            if board.is_drawn(idx):
                print("Invalid move: that line is already drawn. Try again.")
                continue
            return idx

    def choose_action(self, board: Board):
        return board.geom.coords[self.choose_index(board)]
