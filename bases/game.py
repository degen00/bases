"""The ``Bases`` game: a :class:`Board` plus a match loop, logging and stats."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from .board import Board, Transition

log = logging.getLogger("bases")


class Bases(Board):
    """Runs games between two players and keeps a running win tally.

    A player is any object with ``choose_index(board) -> int`` (or the older
    ``choose_action(board) -> (x1, y1, x2, y2)``).  Players that also define
    ``observe(transition)`` are told about every move they make, which is
    how the Q-learning agent learns.
    """

    game_counter = 0

    def __init__(self, size: int, extra_turn_on_box: bool = False,
                 verbose: bool = False, log_moves: bool = False,
                 lines_log_path: Optional[Path] = None):
        super().__init__(size, extra_turn_on_box)
        self.verbose = verbose
        self.log_moves = log_moves
        self.lines_log_path = Path(lines_log_path) if lines_log_path else None
        self.wins = {"A": 0, "B": 0, "Tie": 0}
        self.game_id = 0

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _pick(player, board: Board) -> int:
        if hasattr(player, "choose_index"):
            return player.choose_index(board)
        return board.geom.index_of(player.choose_action(board))

    def print_board(self) -> None:
        print(self.render())

    def print_win_counts(self) -> None:
        print(f"Games won by A: {self.wins['A']}, B: {self.wins['B']}, "
              f"Ties: {self.wins['Tie']}")

    def save_lines_to_csv(self) -> None:
        if self.lines_log_path is None:
            return
        self.lines_log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.lines_log_path.exists()
        with open(self.lines_log_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["game_id", "turn_id", "player", "x1", "y1", "x2", "y2"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
            for row in self.lines:
                writer.writerow({"game_id": self.game_id, **row})

    # -- match loop -------------------------------------------------------
    def play(self, player_a, player_b) -> str:
        """Play one full game. Returns ``'A'``, ``'B'`` or ``'Tie'``."""
        self.reset()
        Bases.game_counter += 1
        self.game_id = Bases.game_counter
        players = {"A": player_a, "B": player_b}
        learners = {p: hasattr(pl, "observe") for p, pl in players.items()}
        log.info("Starting game %d (%dx%d)", self.game_id, self.size, self.size)

        if self.verbose:
            print("\nStarting new game...")
            self.print_win_counts()

        while not self.is_full():
            if self.verbose:
                self.print_board()
            mover = self.turn
            player = players[mover]
            state = self.mask
            idx = self._pick(player, self)
            captured = self.play_index(idx)
            if self.verbose and getattr(player, "name", "human") != "human":
                print(f"{getattr(player, 'name', 'AI')} (Player {mover}) draws "
                      f"{self.geom.coords[idx]}")
            log.debug("Game %d: %s drew %s (+%d)", self.game_id, mover,
                      self.geom.coords[idx], captured)
            if learners[mover]:
                terminal = self.is_full()
                player.observe(Transition(
                    state=state, action=idx, reward=float(captured),
                    next_state=self.mask,
                    next_available=() if terminal else tuple(self.available()),
                    same_mover=self.turn == mover, terminal=terminal,
                    player=mover))

        result = self.winner()
        self.wins[result] += 1
        log.info("Game %d over: %s (A %d - B %d)", self.game_id, result,
                 self.scores["A"], self.scores["B"])
        if self.log_moves:
            self.save_lines_to_csv()
        if self.verbose:
            self.print_board()
            print(f"Game over. Result: {result}  "
                  f"(A {self.scores['A']} - B {self.scores['B']})")
            self.print_win_counts()
        return result
