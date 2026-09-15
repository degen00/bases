"""Kropki (Točki, "Dots"): the free-form base game.

Players alternately place a dot of their colour on an empty grid point.  When
a player's live dots form a closed chain (orthogonal *or diagonal* steps) around
one or more live enemy dots, the enclosed region becomes that player's *base*:
enemy dots inside are captured (one point each) and every point inside is dead
for the rest of the game.  A chain around empty points only is a "house", not
a base; if the opponent later plays inside a house the dot is captured at once.
The board edge never counts as part of a chain.

The game ends when no playable point is left or both players pass in a row;
the player with more captured dots wins.

Rule choices that vary between clubs are constructor flags:

* ``mover_priority`` (default True): when a move completes enclosures for
  both players, the mover's captures are resolved first.
* ``allow_pass`` (default True).

Implementation note: a region is enclosed by player P exactly when it cannot
reach the board edge by 4-connected steps through points that are not P's
live dots (an 8-connected loop of dots is a wall for 4-connected movement).
"""
from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

Point = tuple[int, int]
PLAYERS = ("A", "B")


def other(player: str) -> str:
    return "B" if player == "A" else "A"


class Region:
    """A set of points enclosed by one player's chain."""

    __slots__ = ("owner", "points", "enemy_dots")

    def __init__(self, owner: str, points: list[Point], enemy_dots: list[Point]):
        self.owner = owner
        self.points = points
        self.enemy_dots = enemy_dots

    @property
    def is_base(self) -> bool:
        return bool(self.enemy_dots)


class KropkiBoard:
    def __init__(self, width: int = 10, height: Optional[int] = None,
                 mover_priority: bool = True, allow_pass: bool = True):
        if width < 3 or (height is not None and height < 3):
            raise ValueError("the board needs at least 3x3 points")
        self.width = width
        self.height = height or width
        self.mover_priority = mover_priority
        self.allow_pass = allow_pass
        self.reset()

    def reset(self) -> None:
        w, h = self.width, self.height
        self.dot: list[list[Optional[str]]] = [[None] * w for _ in range(h)]
        self.territory: list[list[Optional[str]]] = [[None] * w for _ in range(h)]
        self.scores = {"A": 0, "B": 0}
        self.turn = "A"
        self.passes = 0
        # (point or None for a pass, player, list of (x, y, old_territory), captured)
        self.history: list[tuple[Optional[Point], str, list[tuple[int, int, Optional[str]]], int]] = []
        self.bases: list[Region] = []

    def copy(self) -> "KropkiBoard":
        clone = KropkiBoard.__new__(KropkiBoard)
        clone.width, clone.height = self.width, self.height
        clone.mover_priority, clone.allow_pass = self.mover_priority, self.allow_pass
        clone.dot = [row[:] for row in self.dot]
        clone.territory = [row[:] for row in self.territory]
        clone.scores = dict(self.scores)
        clone.turn = self.turn
        clone.passes = self.passes
        clone.history = list(self.history)
        clone.bases = list(self.bases)
        return clone

    # -- queries ----------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_live(self, x: int, y: int, player: str) -> bool:
        """A dot of ``player`` that has not been captured."""
        return self.dot[y][x] == player and self.territory[y][x] in (None, player)

    def is_playable(self, x: int, y: int) -> bool:
        return (self.in_bounds(x, y) and self.dot[y][x] is None
                and self.territory[y][x] is None)

    def legal_moves(self) -> list[Point]:
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if self.dot[y][x] is None and self.territory[y][x] is None]

    def is_over(self) -> bool:
        if self.passes >= 2:
            return True
        return not any(self.dot[y][x] is None and self.territory[y][x] is None
                       for y in range(self.height) for x in range(self.width))

    def winner(self) -> str:
        a, b = self.scores["A"], self.scores["B"]
        return "A" if a > b else "B" if b > a else "Tie"

    def margin(self, player: str) -> int:
        return self.scores[player] - self.scores[other(player)]

    def last_move(self) -> Optional[Point]:
        for point, _, _, _ in reversed(self.history):
            return point
        return None

    # -- enclosure detection ----------------------------------------------
    def enclosed_regions(self, player: str) -> list[Region]:
        """Regions the player's live dots wall off from the board edge."""
        w, h = self.width, self.height
        wall = [[self.is_live(x, y, player) for x in range(w)] for y in range(h)]
        reached = [[False] * w for _ in range(h)]
        queue: deque[Point] = deque()
        for x in range(w):
            for y in (0, h - 1):
                if not wall[y][x] and not reached[y][x]:
                    reached[y][x] = True
                    queue.append((x, y))
        for y in range(h):
            for x in (0, w - 1):
                if not wall[y][x] and not reached[y][x]:
                    reached[y][x] = True
                    queue.append((x, y))
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < w and 0 <= ny < h and not wall[ny][nx] and not reached[ny][nx]:
                    reached[ny][nx] = True
                    queue.append((nx, ny))
        regions: list[Region] = []
        seen = [[False] * w for _ in range(h)]
        enemy = other(player)
        for y in range(h):
            for x in range(w):
                if wall[y][x] or reached[y][x] or seen[y][x]:
                    continue
                points: list[Point] = []
                enemies: list[Point] = []
                seen[y][x] = True
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    points.append((cx, cy))
                    if self.is_live(cx, cy, enemy):
                        enemies.append((cx, cy))
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if (0 <= nx < w and 0 <= ny < h and not wall[ny][nx]
                                and not seen[ny][nx]):
                            seen[ny][nx] = True
                            stack.append((nx, ny))
                regions.append(Region(player, points, enemies))
        return regions

    def _claim(self, region: Region, changes: list[tuple[int, int, Optional[str]]]) -> int:
        """Turn a region into ``region.owner``'s base; returns dots captured."""
        captured = 0
        for x, y in region.points:
            if self.territory[y][x] != region.owner:
                changes.append((x, y, self.territory[y][x]))
                self.territory[y][x] = region.owner
        for x, y in region.enemy_dots:
            captured += 1
        self.scores[region.owner] += captured
        self.bases.append(region)
        return captured

    def _resolve(self, player: str, changes) -> int:
        captured = 0
        for region in self.enclosed_regions(player):
            if region.is_base:
                captured += self._claim(region, changes)
        return captured

    # -- moves ------------------------------------------------------------
    def play(self, x: int, y: int) -> tuple[int, int]:
        """Place the current player's dot. Returns (captured by mover,
        captured by opponent). Raises ValueError for an illegal move."""
        if not self.is_playable(x, y):
            raise ValueError(f"({x}, {y}) is not a playable point")
        mover, opponent = self.turn, other(self.turn)
        self.dot[y][x] = mover
        changes: list[tuple[int, int, Optional[str]]] = []
        first, second = (mover, opponent) if self.mover_priority else (opponent, mover)
        got = {first: self._resolve(first, changes)}
        got[second] = self._resolve(second, changes)
        self.history.append(((x, y), mover, changes, got[mover] + got[opponent]))
        self.passes = 0
        self.turn = opponent
        return got[mover], got[opponent]

    def pass_turn(self) -> None:
        if not self.allow_pass:
            raise ValueError("passing is not allowed")
        self.history.append((None, self.turn, [], 0))
        self.passes += 1
        self.turn = other(self.turn)

    def undo(self) -> bool:
        if not self.history:
            return False
        point, player, changes, _ = self.history.pop()
        if point is None:
            self.passes -= 1
        else:
            x, y = point
            self.dot[y][x] = None
            for cx, cy, old in reversed(changes):
                self.territory[cy][cx] = old
            self.passes = 0
            for prev in reversed(self.history):
                if prev[0] is not None:
                    break
                self.passes += 1
        self.turn = player
        self.scores = self._recount()
        self.bases = [b for b in self.bases if all(
            self.territory[y][x] == b.owner for x, y in b.points)]
        return True

    def _recount(self) -> dict[str, int]:
        scores = {"A": 0, "B": 0}
        for y in range(self.height):
            for x in range(self.width):
                d, t = self.dot[y][x], self.territory[y][x]
                if d is not None and t is not None and t != d:
                    scores[t] += 1
        return scores

    # -- rendering --------------------------------------------------------
    def render(self) -> str:
        """ASCII board: A/B live dots, a/b captured dots, '.' empty,
        '+'/'x' empty points inside A's/B's bases."""
        header = "   " + " ".join(f"{x % 10}" for x in range(self.width))
        rows = [header]
        for y in range(self.height):
            cells = []
            for x in range(self.width):
                d, t = self.dot[y][x], self.territory[y][x]
                if d is None:
                    cells.append("." if t is None else ("+" if t == "A" else "x"))
                elif t is None or t == d:
                    cells.append(d)
                else:
                    cells.append(d.lower())
            rows.append(f"{y:>2} " + " ".join(cells))
        rows.append(f"   A: {self.scores['A']}  B: {self.scores['B']}  to move: {self.turn}")
        return "\n".join(rows)

    def __str__(self) -> str:
        return self.render()
