"""Self-play matches between Kropki AI levels: the tuning and regression tool.

    python play.py kropki-eval --levels random,normal,hard --games 4 --width 8

Each pair plays ``games`` games with the colours swapped every game so that
first-move advantage cancels out.  Results report wins from the perspective
of the first-named level, the average margin and the time per move.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Optional

from .kropki import KropkiBoard
from .kropki_ai import make_kropki_ai

Factory = Callable[[Optional[int]], object]


@dataclass
class MatchResult:
    left: str
    right: str
    games: int
    wins: int = 0
    losses: int = 0
    ties: int = 0
    margin_sum: int = 0
    ms_left: float = 0.0
    ms_right: float = 0.0
    scores: list[tuple[int, int]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.ties) / self.games if self.games else 0.0

    @property
    def avg_margin(self) -> float:
        return self.margin_sum / self.games if self.games else 0.0

    def __str__(self) -> str:
        return (f"{self.left:>8} vs {self.right:<8} {self.wins}W {self.losses}L {self.ties}T "
                f"score {self.score:.2f}  margin {self.avg_margin:+.1f}  "
                f"{self.ms_left:.0f} / {self.ms_right:.0f} ms/move")


def play_game(player_a, player_b, width: int, height: int,
              max_moves: Optional[int] = None) -> tuple[KropkiBoard, dict[str, float]]:
    """One game; returns the final board and seconds spent per player."""
    board = KropkiBoard(width, height)
    spent = {"A": 0.0, "B": 0.0}
    counts = {"A": 0, "B": 0}
    players = {"A": player_a, "B": player_b}
    while not board.is_over():
        if max_moves is not None and len(board.history) >= max_moves:
            break
        side = board.turn
        t0 = time.perf_counter()
        move = players[side].choose_move(board)
        spent[side] += time.perf_counter() - t0
        counts[side] += 1
        if move is None:
            board.pass_turn()
        else:
            board.play(*move)
    per_move = {s: (spent[s] / counts[s] if counts[s] else 0.0) for s in "AB"}
    return board, per_move


def play_match(left: str, right: str, games: int, width: int, height: int,
               seed: Optional[int] = None, factory: Callable[[str, Optional[int]], object] = make_kropki_ai,
               verbose: bool = False) -> MatchResult:
    rng = random.Random(seed)
    result = MatchResult(left, right, games)
    for i in range(games):
        left_is_a = i % 2 == 0
        pl = factory(left, rng.randrange(1 << 30))
        pr = factory(right, rng.randrange(1 << 30))
        board, per_move = play_game(pl if left_is_a else pr, pr if left_is_a else pl, width, height)
        side = "A" if left_is_a else "B"
        margin = board.margin(side)
        result.margin_sum += margin
        result.scores.append((board.scores[side], board.scores["B" if side == "A" else "A"]))
        if margin > 0:
            result.wins += 1
        elif margin < 0:
            result.losses += 1
        else:
            result.ties += 1
        result.ms_left += per_move[side] * 1000 / games
        result.ms_right += per_move["B" if side == "A" else "A"] * 1000 / games
        if verbose:
            print(f"  game {i + 1}: {left} as {side} -> {board.scores[side]} : "
                  f"{board.scores['B' if side == 'A' else 'A']}", flush=True)
    return result


def tournament(levels: list[str], games: int, width: int, height: int,
               seed: Optional[int] = None, verbose: bool = False) -> list[MatchResult]:
    results = []
    for left, right in combinations(levels, 2):
        if verbose:
            print(f"{left} vs {right}:", flush=True)
        results.append(play_match(left, right, games, width, height, seed, verbose=verbose))
        print(results[-1], flush=True)
    return results
