"""Computer players for Kropki.

A tabular Q-learner cannot cover this game (a 10x10 board has ~3^100
positions and no useful symmetry reduction), so these players search:

* :class:`RandomKropki` - uniformly random legal move.
* :class:`HeuristicKropki` - one-ply evaluation of every legal move
  (captures, dots saved from capture, self-atari avoidance, proximity to
  enemy dots, connectivity, centre bias) with optional random ``noise``.
* :class:`SearchKropki` - the same evaluation, but the best ``breadth``
  candidates are checked against the opponent's best reply (2-ply minimax).
"""
from __future__ import annotations

import random
from typing import Optional

from .kropki import KropkiBoard, Point, other

NEIGHBOURS8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


class RandomKropki:
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: KropkiBoard) -> Optional[Point]:
        moves = board.legal_moves()
        return self.rng.choice(moves) if moves else None


def capture_delta(board: KropkiBoard, move: Point) -> tuple[int, int]:
    """(dots the mover captures, dots the mover loses) if ``move`` is played."""
    trial = board.copy()
    mine, theirs = trial.play(*move)
    return mine, theirs


def threat(board: KropkiBoard, move: Point) -> int:
    """Dots the *opponent* would capture by playing ``move`` right now."""
    trial = board.copy()
    trial.turn = other(board.turn)
    mine, _ = trial.play(*move)
    return mine


class Group:
    """A 4-connected group of one player's live dots with its liberties."""

    __slots__ = ("player", "dots", "liberties", "touches_edge")

    def __init__(self, player, dots, liberties, touches_edge):
        self.player = player
        self.dots = dots
        self.liberties = liberties
        self.touches_edge = touches_edge


def groups4(board: KropkiBoard, player: str) -> list[Group]:
    """Orthogonally connected groups of ``player``'s live dots.

    A group is captured exactly when every orthogonal liberty holds an enemy
    live dot (those points form a closed diagonal chain), unless it touches
    the board edge, where no chain can close around it.
    """
    w, h = board.width, board.height
    seen = [[False] * w for _ in range(h)]
    out = []
    for y0 in range(h):
        for x0 in range(w):
            if seen[y0][x0] or not board.is_live(x0, y0, player):
                continue
            dots, libs, edge = [], set(), False
            seen[y0][x0] = True
            stack = [(x0, y0)]
            while stack:
                x, y = stack.pop()
                dots.append((x, y))
                if x == 0 or y == 0 or x == w - 1 or y == h - 1:
                    edge = True
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if board.is_live(nx, ny, player):
                        if not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((nx, ny))
                    elif board.is_playable(nx, ny):
                        libs.add((nx, ny))
            out.append(Group(player, dots, libs, edge))
    return out


def danger(board: KropkiBoard, player: str) -> float:
    """How exposed ``player``'s interior groups are (bigger = worse)."""
    total = 0.0
    for g in groups4(board, player):
        if g.touches_edge:
            continue
        total += len(g.dots) / max(1, len(g.liberties))
    return total


class HeuristicKropki:
    name = "heuristic"

    W_CAPTURE = 10.0
    W_LOSS = 10.0
    W_BLOCK = 8.0
    W_ENEMY_NEAR = 1.2
    W_FRIEND_NEAR = 0.6
    W_CENTRE = 0.15
    W_LIB = 2.5            # liberty pressure on interior enemy groups
    W_ATARI = 4.0          # leaving an interior enemy group with one liberty
    W_ESCAPE = 3.0         # per liberty gained by an own group in danger

    def __init__(self, seed: Optional[int] = None, noise: float = 0.0):
        self.rng = random.Random(seed)
        self.noise = noise

    def evaluate(self, board: KropkiBoard, move: Point, quick: bool = False,
                 context: Optional[tuple] = None) -> float:
        """Static score of ``move`` for the player to move.  ``context`` is
        the precomputed ``(enemy groups, own groups)`` from :meth:`ranked_moves`."""
        me, enemy = board.turn, other(board.turn)
        trial = board.copy()
        mine, theirs = trial.play(*move)
        score = self.W_CAPTURE * mine - self.W_LOSS * theirs
        if mine == 0:
            score += self.W_BLOCK * threat(board, move)
        enemy_groups, own_groups = context or (groups4(board, enemy), groups4(board, me))
        if theirs == 0:
            for g in enemy_groups:
                if g.touches_edge or move not in g.liberties:
                    continue
                left = len(g.liberties) - 1
                score += self.W_LIB * len(g.dots) / max(1, left)
                if left == 1:
                    score += self.W_ATARI * len(g.dots)
            x, y = move
            new_libs = {(nx, ny) for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                        if board.is_playable(nx, ny)}
            for g in own_groups:
                if g.touches_edge or move not in g.liberties or len(g.liberties) > 2:
                    continue
                gained = len((g.liberties | new_libs) - {move}) - len(g.liberties)
                score += self.W_ESCAPE * gained * len(g.dots)
        x, y = move
        enemies = friends = 0
        for dx, dy in NEIGHBOURS8:
            nx, ny = x + dx, y + dy
            if board.in_bounds(nx, ny):
                if board.is_live(nx, ny, enemy):
                    enemies += 1
                elif board.is_live(nx, ny, me):
                    friends += 1
        score += self.W_ENEMY_NEAR * min(enemies, 2) + self.W_FRIEND_NEAR * min(friends, 2)
        cx, cy = (board.width - 1) / 2, (board.height - 1) / 2
        score -= self.W_CENTRE * (abs(x - cx) + abs(y - cy))
        return score

    def ranked_moves(self, board: KropkiBoard, quick: bool = False) -> list[tuple[float, Point]]:
        context = (groups4(board, other(board.turn)), groups4(board, board.turn))
        ranked = [(self.evaluate(board, m, quick, context), m) for m in board.legal_moves()]
        self.rng.shuffle(ranked)          # random tie-breaking
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def choose_move(self, board: KropkiBoard) -> Optional[Point]:
        moves = board.legal_moves()
        if not moves:
            return None
        if self.noise > 0 and self.rng.random() < self.noise:
            return self.rng.choice(moves)
        return self.ranked_moves(board)[0][1]


def best_capture(board: KropkiBoard, player: str) -> int:
    """Largest capture ``player`` could make if it were their move."""
    best = 0
    trial = board.copy()
    trial.turn = player
    for move in trial.legal_moves():
        t = trial.copy()
        mine, _ = t.play(*move)
        if mine > best:
            best = mine
    return best


class SearchKropki(HeuristicKropki):
    """Two-ply minimax over the ``breadth`` best heuristic moves per side.

    Leaves are scored with the box margin plus a threat term: the largest
    capture the side to move could take next, minus the largest capture the
    other side is threatening.  This sees forks and forced captures three
    plies deep at modest cost.
    """

    name = "search"
    W_THREAT = 6.0

    def __init__(self, seed: Optional[int] = None, breadth: int = 6, noise: float = 0.0):
        super().__init__(seed, noise)
        self.breadth = breadth

    def leaf_value(self, board: KropkiBoard, me: str) -> float:
        value = self.W_CAPTURE * board.margin(me)
        if board.is_over():
            return value
        enemy = other(me)
        return (value + self.W_THREAT * best_capture(board, me)
                - self.W_THREAT * best_capture(board, enemy)
                + self.W_LIB * (danger(board, enemy) - danger(board, me)))

    def choose_move(self, board: KropkiBoard) -> Optional[Point]:
        moves = board.legal_moves()
        if not moves:
            return None
        if self.noise > 0 and self.rng.random() < self.noise:
            return self.rng.choice(moves)
        candidates = self.ranked_moves(board)[:self.breadth]
        if len(candidates) == 1:
            return candidates[0][1]
        me = board.turn
        best_value, best_move = None, candidates[0][1]
        for static, move in candidates:
            trial = board.copy()
            trial.play(*move)
            if trial.is_over():
                value = self.leaf_value(trial, me)
            else:
                value = None
                for _, reply in self.ranked_moves(trial, quick=True)[:self.breadth]:
                    t2 = trial.copy()
                    t2.play(*reply)
                    leaf = self.leaf_value(t2, me)
                    if value is None or leaf < value:
                        value = leaf
            value += 0.05 * static                      # positional tie-break
            if best_value is None or value > best_value:
                best_value, best_move = value, move
        return best_move


KROPKI_AI = {"random": RandomKropki, "heuristic": HeuristicKropki, "search": SearchKropki}


def make_kropki_ai(difficulty: str, seed: Optional[int] = None):
    """easy: heuristic with 35% random moves; normal: heuristic; hard: search."""
    if difficulty == "easy":
        return HeuristicKropki(seed, noise=0.35)
    if difficulty == "hard":
        return SearchKropki(seed)
    return HeuristicKropki(seed)
