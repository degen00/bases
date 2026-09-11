"""Dots-and-Boxes board engine.

The set of drawn lines is stored as the bits of one integer (``Board.mask``).
Everything the agents need that depends only on the grid size (which boxes a
line touches, the eight dihedral symmetries as permutations of line indices,
...) is precomputed once per size in :class:`Geometry`.

Coordinates follow the original text interface: a dot is ``(x, y)`` with
``x`` the column and ``y`` the row, ``(0, 0)`` top-left.  A *move* is the
4-tuple ``(x1, y1, x2, y2)`` of two adjacent dots.  Internally every line also
has an integer index in ``range(Geometry.num_lines)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

Move = tuple[int, int, int, int]
PLAYERS = ("A", "B")


def other(player: str) -> str:
    return "B" if player == "A" else "A"


class Geometry:
    """Static lookup tables for a ``size`` x ``size`` grid of boxes."""

    _cache: dict[int, "Geometry"] = {}

    @classmethod
    def get(cls, size: int) -> "Geometry":
        geom = cls._cache.get(size)
        if geom is None:
            geom = cls._cache[size] = cls(size)
        return geom

    def __init__(self, size: int):
        if size < 1:
            raise ValueError("size must be at least 1")
        n = self.size = size
        self.num_boxes = n * n
        self.num_lines = 2 * n * (n + 1)

        # Horizontal lines first (row major), then vertical lines.
        self.coords: list[Move] = []
        for r in range(n + 1):
            for c in range(n):
                self.coords.append((c, r, c + 1, r))
        for r in range(n):
            for c in range(n + 1):
                self.coords.append((c, r, c, r + 1))
        self.index: dict[Move, int] = {m: i for i, m in enumerate(self.coords)}

        # Box <-> line adjacency.  Box ``b = row * n + col``.
        self.box_sides: list[tuple[int, int, int, int]] = []
        self.line_boxes: list[tuple[int, ...]] = [() for _ in range(self.num_lines)]
        boxes_of_line: list[list[int]] = [[] for _ in range(self.num_lines)]
        for r in range(n):
            for c in range(n):
                b = r * n + c
                sides = (self.h(r, c), self.h(r + 1, c), self.v(r, c), self.v(r, c + 1))
                self.box_sides.append(sides)
                for s in sides:
                    boxes_of_line[s].append(b)
        self.line_boxes = [tuple(bs) for bs in boxes_of_line]
        self.box_masks = [sum(1 << s for s in sides) for sides in self.box_sides]
        self.full_mask = (1 << self.num_lines) - 1

        # The eight symmetries of the square as permutations of line indices.
        # ``perms[0]`` is the identity.
        self.perms: list[list[int]] = self._build_symmetries()
        self.inverse: list[list[int]] = [self._invert(p) for p in self.perms]
        # Byte-wise lookup tables so a whole mask is transformed with
        # ``ceil(num_lines / 8)`` table lookups instead of one loop per bit.
        self._tables: list[list[list[int]]] = [self._byte_tables(p) for p in self.perms]

    # -- indexing helpers -------------------------------------------------
    def h(self, row: int, col: int) -> int:
        """Index of the horizontal line above box (row, col)."""
        return row * self.size + col

    def v(self, row: int, col: int) -> int:
        """Index of the vertical line left of box (row, col)."""
        return self.size * (self.size + 1) + row * (self.size + 1) + col

    @staticmethod
    def normalise(move: Move) -> Move:
        x1, y1, x2, y2 = move
        if (x1, y1) > (x2, y2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        return (x1, y1, x2, y2)

    def index_of(self, move: Move) -> int:
        try:
            return self.index[self.normalise(tuple(move))]
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"{tuple(move)} is not a line of a {self.size}x{self.size} board"
            ) from None

    # -- symmetries -------------------------------------------------------
    def _build_symmetries(self) -> list[list[int]]:
        n = self.size

        def rot(p: tuple[int, int]) -> tuple[int, int]:
            x, y = p
            return (n - y, x)

        def refl(p: tuple[int, int]) -> tuple[int, int]:
            x, y = p
            return (n - x, y)

        maps: list[Callable[[tuple[int, int]], tuple[int, int]]] = []
        for flip in (False, True):
            for k in range(4):
                def f(p, k=k, flip=flip):
                    for _ in range(k):
                        p = rot(p)
                    return refl(p) if flip else p
                maps.append(f)

        perms = []
        for f in maps:
            perm = []
            for (x1, y1, x2, y2) in self.coords:
                a, b = f((x1, y1)), f((x2, y2))
                perm.append(self.index_of((a[0], a[1], b[0], b[1])))
            perms.append(perm)
        return perms

    @staticmethod
    def _invert(perm: list[int]) -> list[int]:
        inv = [0] * len(perm)
        for i, j in enumerate(perm):
            inv[j] = i
        return inv

    def _byte_tables(self, perm: list[int]) -> list[list[int]]:
        nbytes = (self.num_lines + 7) // 8
        tables = []
        for j in range(nbytes):
            table = []
            for byte in range(256):
                out = 0
                for bit in range(8):
                    i = j * 8 + bit
                    if byte >> bit & 1 and i < self.num_lines:
                        out |= 1 << perm[i]
                table.append(out)
            tables.append(table)
        return tables

    def transform(self, mask: int, k: int) -> int:
        """Apply symmetry ``k`` (index into ``perms``) to a bitmask of lines."""
        out = 0
        for j, table in enumerate(self._tables[k]):
            out |= table[(mask >> (8 * j)) & 255]
        return out

    @staticmethod
    def permute(mask: int, perm: list[int]) -> int:
        """Apply an arbitrary line permutation to a bitmask (slow path)."""
        out = 0
        while mask:
            low = mask & -mask
            out |= 1 << perm[low.bit_length() - 1]
            mask ^= low
        return out

    def prior_row(self, mask: int) -> tuple[float, ...]:
        """Heuristic value of every line in position ``mask``: boxes the line
        completes minus boxes it leaves with three sides (0 for drawn lines)."""
        return _prior_row(self.size, mask)

    def canonical(self, mask: int) -> tuple[int, tuple[int, ...]]:
        """Return ``(canonical_mask, ks)``: the numerically smallest mask of
        the symmetry class and every symmetry index ``k`` that reaches it.

        ``ks`` has one element unless the position is itself symmetric, in
        which case an action must be canonicalised with :meth:`canonical_action`.
        """
        return _canonical(self.size, mask)

    def canonical_action(self, ks: tuple[int, ...], action: int) -> int:
        """Map a line index into the canonical frame given by ``ks``."""
        if len(ks) == 1:
            return self.perms[ks[0]][action]
        return min(self.perms[k][action] for k in ks)


@lru_cache(maxsize=1 << 20)
def _prior_row(size: int, mask: int) -> tuple[float, ...]:
    geom = Geometry.get(size)
    sides = [(mask & bm).bit_count() for bm in geom.box_masks]
    row = []
    for i in range(geom.num_lines):
        if mask >> i & 1:
            row.append(0.0)
            continue
        value = 0
        for b in geom.line_boxes[i]:
            if sides[b] == 3:
                value += 1
            elif sides[b] == 2:
                value -= 1
        row.append(float(value))
    return tuple(row)


@lru_cache(maxsize=1 << 20)
def _canonical(size: int, mask: int) -> tuple[int, tuple[int, ...]]:
    geom = Geometry.get(size)
    best, ks = mask, [0]
    for k in range(1, 8):
        out = 0
        for j, table in enumerate(geom._tables[k]):
            out |= table[(mask >> (8 * j)) & 255]
        if out < best:
            best, ks = out, [k]
        elif out == best:
            ks.append(k)
    return best, tuple(ks)


@dataclass(frozen=True)
class Transition:
    """One move as seen by a learning agent.

    ``reward`` is the number of boxes the mover captured, ``same_mover`` tells
    whether the mover also plays next (extra-turn rule), and ``terminal``
    whether the game ended with this move.
    """
    state: int
    action: int
    reward: float
    next_state: int
    next_available: tuple[int, ...]
    same_mover: bool
    terminal: bool
    player: str


class Board:
    """Game state: drawn lines, box ownership, scores and whose turn it is."""

    def __init__(self, size: int, extra_turn_on_box: bool = False):
        self.geom = Geometry.get(size)
        self.size = size
        self.extra_turn_on_box = extra_turn_on_box
        self.reset()

    def reset(self) -> None:
        self.mask = 0
        self._avail_cache: tuple[int, list[int]] = (-1, [])
        self.owner: list[Optional[str]] = [None] * self.geom.num_boxes
        self.scores = {"A": 0, "B": 0}
        self.turn = "A"
        # (line index, player, captured boxes) per move, for undo and logging
        self.history: list[tuple[int, str, tuple[int, ...]]] = []

    def copy(self) -> "Board":
        clone = Board.__new__(Board)
        clone.geom = self.geom
        clone.size = self.size
        clone.extra_turn_on_box = self.extra_turn_on_box
        clone.mask = self.mask
        clone._avail_cache = (-1, [])
        clone.owner = list(self.owner)
        clone.scores = dict(self.scores)
        clone.turn = self.turn
        clone.history = list(self.history)
        return clone

    # -- queries ----------------------------------------------------------
    def is_drawn(self, idx: int) -> bool:
        return (self.mask >> idx) & 1 == 1

    def is_line_drawn(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        try:
            return self.is_drawn(self.geom.index_of((x1, y1, x2, y2)))
        except ValueError:
            return False

    def available(self) -> list[int]:
        """Indices of undrawn lines (a fresh list each call)."""
        mask = self.mask
        if self._avail_cache[0] != mask:
            self._avail_cache = (mask, [i for i in range(self.geom.num_lines)
                                        if not (mask >> i) & 1])
        return list(self._avail_cache[1])

    def available_moves(self) -> list[Move]:
        coords = self.geom.coords
        return [coords[i] for i in self.available()]

    def is_full(self) -> bool:
        return self.mask == self.geom.full_mask

    is_over = is_full

    def sides(self, box: int) -> int:
        """Number of drawn sides of box ``box``."""
        return (self.mask & self.geom.box_masks[box]).bit_count()

    def captures_for(self, idx: int) -> int:
        """Boxes that drawing line ``idx`` would complete."""
        return sum(1 for b in self.geom.line_boxes[idx] if self.sides(b) == 3)

    def gifts_for(self, idx: int) -> int:
        """Boxes that drawing line ``idx`` would leave with three sides."""
        return sum(1 for b in self.geom.line_boxes[idx] if self.sides(b) == 2)

    def winner(self) -> str:
        a, b = self.scores["A"], self.scores["B"]
        return "A" if a > b else "B" if b > a else "Tie"

    def margin(self, player: str) -> int:
        return self.scores[player] - self.scores[other(player)]

    @property
    def boxes(self) -> list[list[Optional[str]]]:
        n = self.size
        return [[self.owner[r * n + c] for c in range(n)] for r in range(n)]

    @property
    def lines(self) -> list[dict]:
        rows = []
        for turn_id, (idx, player, _) in enumerate(self.history, start=1):
            x1, y1, x2, y2 = self.geom.coords[idx]
            rows.append({"turn_id": turn_id, "player": player,
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return rows

    # -- mutation ---------------------------------------------------------
    def play_index(self, idx: int) -> int:
        """Draw line ``idx`` for the current player. Returns boxes captured."""
        if not 0 <= idx < self.geom.num_lines:
            raise ValueError(f"line index {idx} out of range")
        if self.is_drawn(idx):
            raise ValueError("line already drawn")
        self.mask |= 1 << idx
        captured = []
        for b in self.geom.line_boxes[idx]:
            bm = self.geom.box_masks[b]
            if self.owner[b] is None and self.mask & bm == bm:
                self.owner[b] = self.turn
                captured.append(b)
        self.scores[self.turn] += len(captured)
        self.history.append((idx, self.turn, tuple(captured)))
        if not (captured and self.extra_turn_on_box):
            self.turn = other(self.turn)
        return len(captured)

    def make_move(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Draw the line between two adjacent dots. Returns boxes captured.

        Raises ``ValueError`` for an illegal move.
        """
        return self.play_index(self.geom.index_of((x1, y1, x2, y2)))

    def undo(self) -> bool:
        """Take back the last move. Returns False if there is nothing to undo."""
        if not self.history:
            return False
        idx, player, captured = self.history.pop()
        self.mask &= ~(1 << idx)
        for b in captured:
            self.owner[b] = None
        self.scores[player] -= len(captured)
        self.turn = player
        return True

    # -- rendering --------------------------------------------------------
    def render(self) -> str:
        n = self.size
        delimiter = "-" * (n * 4 + 1)
        out = [delimiter]
        for row in range(n + 1):
            line = ""
            for col in range(n):
                line += "*" + ("---" if self.is_line_drawn(col, row, col + 1, row) else "   ")
            out.append(line + "*")
            if row == n:
                break
            line = ""
            for col in range(n + 1):
                line += "|" if self.is_line_drawn(col, row, col, row + 1) else " "
                if col < n:
                    owner = self.owner[row * n + col]
                    line += f" {owner or ' '} "
            out.append(line)
        out.append(delimiter)
        return "\n".join(out)

    def __str__(self) -> str:
        return self.render()
