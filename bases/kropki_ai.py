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
import time
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


class SearchTimeout(Exception):
    """Raised inside the search when the time budget is exhausted."""


def capture_map(board: KropkiBoard, player: str) -> tuple[dict[Point, int], set[Point]]:
    """Exact capturing moves for ``player`` without playing anything.

    Consider the graph of *open* points (everything but ``player``'s live
    dots, 4-connected) plus a virtual EDGE node joined to every open border
    point.  Placing a dot at ``p`` removes ``p`` from that graph, so it
    captures exactly the enemy dots whose component is thereby cut off from
    EDGE: ``p`` must be an articulation point, and the captured dots are those
    in the DFS subtrees it separates.  One iterative Tarjan pass yields, for
    every playable point, the number of enemy dots it would capture.

    Returns ``(captures, reachable)`` where ``captures[p]`` is that number for
    each playable ``p`` with a capture, and ``reachable`` is the set of open
    points connected to the edge (playable points outside it lie inside one
    of ``player``'s houses).
    """
    w, h = board.width, board.height
    enemy = other(player)
    open_ = [[not board.is_live(x, y, player) for x in range(w)] for y in range(h)]
    EDGE = (-1, -1)

    def neighbours(node):
        if node is EDGE:
            for x in range(w):
                if open_[0][x]:
                    yield (x, 0)
                if h > 1 and open_[h - 1][x]:
                    yield (x, h - 1)
            for y in range(1, h - 1):
                if open_[y][0]:
                    yield (0, y)
                if w > 1 and open_[y][w - 1]:
                    yield (w - 1, y)
            return
        x, y = node
        if x == 0 or y == 0 or x == w - 1 or y == h - 1:
            yield EDGE
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and open_[ny][nx]:
                yield (nx, ny)

    disc: dict = {EDGE: 0}
    low: dict = {EDGE: 0}
    sub: dict = {EDGE: 0}
    captures: dict[Point, int] = {}
    counter = 1
    stack = [(EDGE, None, neighbours(EDGE))]
    while stack:
        node, parent, it = stack[-1]
        advanced = False
        for nxt in it:
            if nxt == parent:
                continue
            if nxt in disc:
                if disc[nxt] < low[node]:
                    low[node] = disc[nxt]
                continue
            disc[nxt] = low[nxt] = counter
            counter += 1
            x, y = nxt
            sub[nxt] = 1 if board.is_live(x, y, enemy) else 0
            stack.append((nxt, node, neighbours(nxt)))
            advanced = True
            break
        if advanced:
            continue
        stack.pop()
        if parent is not None:
            if low[node] < low[parent]:
                low[parent] = low[node]
            sub[parent] += sub[node]
            if parent is not EDGE and low[node] >= disc[parent] and sub[node]:
                captures[parent] = captures.get(parent, 0) + sub[node]
    reachable = set(disc)
    reachable.discard(EDGE)
    playable = {p: n for p, n in captures.items() if board.is_playable(*p)}
    return playable, reachable


class QuickContext:
    """Everything the trial-free move scoring needs: exact capture maps for
    both sides (see :func:`capture_map`) and liberty maps for pressure."""

    __slots__ = ("me", "enemy", "enemy_at", "own_at", "my_caps", "their_caps",
                 "my_houses_reach", "their_houses_reach")

    def __init__(self, board: KropkiBoard):
        self.me, self.enemy = board.turn, other(board.turn)
        self.my_caps, self.my_houses_reach = capture_map(board, self.me)
        self.their_caps, self.their_houses_reach = capture_map(board, self.enemy)
        self.enemy_at: dict[Point, list[Group]] = {}
        self.own_at: dict[Point, list[Group]] = {}
        for g in groups4(board, self.enemy):
            if not g.touches_edge:
                for p in g.liberties:
                    self.enemy_at.setdefault(p, []).append(g)
        for g in groups4(board, self.me):
            for p in g.liberties:
                self.own_at.setdefault(p, []).append(g)

    def in_enemy_house(self, p: Point) -> bool:
        """Playable point enclosed by the enemy: a dot placed there is lost."""
        return p not in self.their_houses_reach


def board_key(board: KropkiBoard) -> tuple:
    """Hashable position identity for the transposition table."""
    return (board.turn,
            tuple("".join(c or "." for c in row) for row in board.dot),
            tuple("".join(c or "." for c in row) for row in board.territory))


class SearchKropki(HeuristicKropki):
    """Iterative-deepening alpha-beta over a heuristic beam, with a per-move
    time budget and a transposition table.

    Interior nodes rank moves with :meth:`quick_parts`: captures, blocks and
    house points come from :func:`capture_map` (exact, one graph pass per
    side, no trial plays), liberties add pressure/escape terms; the best
    ``verify`` candidates are then confirmed with a real play so that
    mover-priority corner cases are exact too.  Leaves (:meth:`quick_leaf`)
    score the dot margin, the largest capture each side can make next and how
    endangered each side's interior groups are.  ``time_budget=None``
    searches a fixed ``max_depth`` (deterministic; used by the tests).
    """

    name = "search"
    W_THREAT = 6.0

    def __init__(self, seed: Optional[int] = None, breadth: int = 6, noise: float = 0.0,
                 max_depth: int = 2, time_budget: Optional[float] = None,
                 root_breadth: Optional[int] = None, verify: int = 8):
        super().__init__(seed, noise)
        self.breadth = breadth
        self.root_breadth = root_breadth or max(breadth, 8)
        self.verify = verify          # candidates verified with a real play per node
        self.max_depth = max_depth
        self.time_budget = time_budget
        self.tt: dict[tuple, tuple[float, int, Optional[Point]]] = {}
        self.nodes = 0
        self.depth_reached = 0

    # -- cheap evaluation ---------------------------------------------------
    def quick_score(self, board: KropkiBoard, move: Point, ctx: QuickContext) -> float:
        base, capture = self.quick_parts(board, move, ctx)
        return base + self.W_CAPTURE * capture

    def quick_parts(self, board: KropkiBoard, move: Point, ctx: QuickContext) -> tuple[float, int]:
        """(positional score, estimated capture) for ``move`` without a trial
        play.  The capture estimate comes from liberties and is verified with
        a real play for the best candidates (see :meth:`exact_candidates`)."""
        x, y = move
        score = 0.0
        capture = ctx.my_caps.get(move, 0)               # exact (articulation analysis)
        for g in ctx.enemy_at.get(move, ()):
            left = len(g.liberties) - 1
            if left > 0:
                score += self.W_LIB * len(g.dots) / left
                if left == 1:
                    score += self.W_ATARI * len(g.dots)
        own_here = ctx.own_at.get(move, ())
        if capture == 0:
            score += self.W_BLOCK * ctx.their_caps.get(move, 0)   # exact block value
            if ctx.in_enemy_house(move):
                score -= self.W_LOSS                              # dot would be lost at once
            libs = {(nx, ny) for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                    if board.is_playable(nx, ny)}
            safe = x == 0 or y == 0 or x == board.width - 1 or y == board.height - 1
            for g in own_here:
                libs |= g.liberties
                safe = safe or g.touches_edge
            libs.discard(move)
            if not safe:
                if len(libs) == 1:
                    score -= 0.5 * self.W_LOSS
                for g in own_here:
                    if len(g.liberties) <= 2:
                        score += self.W_ESCAPE * (len(libs) - len(g.liberties)) * len(g.dots)
        enemies = friends = 0
        for dx, dy in NEIGHBOURS8:
            nx, ny = x + dx, y + dy
            if board.in_bounds(nx, ny):
                if board.is_live(nx, ny, ctx.enemy):
                    enemies += 1
                elif board.is_live(nx, ny, ctx.me):
                    friends += 1
        score += self.W_ENEMY_NEAR * min(enemies, 2) + self.W_FRIEND_NEAR * min(friends, 2)
        cx, cy = (board.width - 1) / 2, (board.height - 1) / 2
        score -= self.W_CENTRE * (abs(x - cx) + abs(y - cy))
        return score, capture

    def exact_candidates(self, board: KropkiBoard, k: int
                         ) -> list[tuple[float, Point, KropkiBoard, int, int]]:
        """The ``k`` best moves for the side to move, pre-ranked by
        :meth:`quick_parts` and then verified with a real play so that
        captures and losses are exact.  Returns ``(score, move, child, mine,
        theirs)`` sorted best first; ``child`` is the position after the move."""
        ctx = QuickContext(board)
        pre = []
        for m in board.legal_moves():
            base, est = self.quick_parts(board, m, ctx)
            pre.append((base + self.W_CAPTURE * est, base, m))
        pre.sort(key=lambda item: item[0], reverse=True)
        out = []
        for _, base, m in pre[:max(k, self.verify)]:
            child = board.copy()
            mine, theirs = child.play(*m)
            out.append((base + self.W_CAPTURE * mine - self.W_LOSS * theirs, m, child, mine, theirs))
        out.sort(key=lambda item: item[0], reverse=True)
        return out[:k]

    def quick_leaf(self, board: KropkiBoard, me: str) -> float:
        """Static value of ``board`` for ``me`` (who is to move at the leaf):
        dot margin, the largest capture each side can make next (exact, from
        the articulation analysis) and how endangered interior groups are."""
        value = self.W_CAPTURE * board.margin(me)
        if board.is_over():
            return value
        enemy = other(me)
        my_caps, _ = capture_map(board, me)
        their_caps, _ = capture_map(board, enemy)
        my_threat = max(my_caps.values(), default=0)
        their_threat = max(their_caps.values(), default=0)
        return (value + self.W_THREAT * (my_threat - their_threat)
                + self.W_LIB * (danger(board, enemy) - danger(board, me)))

    # -- search -------------------------------------------------------------
    def _negamax(self, board: KropkiBoard, depth: int, alpha: float, beta: float,
                 deadline: Optional[float]) -> float:
        self.nodes += 1
        if deadline is not None and time.monotonic() > deadline:
            raise SearchTimeout
        if depth == 0 or board.is_over():
            return self.quick_leaf(board, board.turn)
        key = (board_key(board), depth)
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            value, flag, tt_move = entry
            if flag == 0:
                return value
            if flag == 1:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value
        beam = [(m, child) for _, m, child, _, _ in self.exact_candidates(board, self.breadth)]
        if tt_move is not None:
            for i, (m, _) in enumerate(beam):
                if m == tt_move and i:
                    beam.insert(0, beam.pop(i))
                    break
        alpha0 = alpha
        best, best_move = -INF, None
        for move, child in beam:
            value = -self._negamax(child, depth - 1, -beta, -alpha, deadline)
            if value > best:
                best, best_move = value, move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        flag = 0 if alpha0 < best < beta else (2 if best <= alpha0 else 1)
        self.tt[key] = (best, flag, best_move)
        return best

    def choose_move(self, board: KropkiBoard) -> Optional[Point]:
        moves = board.legal_moves()
        if not moves:
            return None
        if self.noise > 0 and self.rng.random() < self.noise:
            return self.rng.choice(moves)
        root = self.ranked_moves(board)[:self.root_breadth]      # exact static ordering
        order = [m for _, m in root]
        static = {m: s for s, m in root}
        if len(order) == 1:
            return order[0]
        deadline = time.monotonic() + self.time_budget if self.time_budget else None
        self.tt = {}
        self.nodes = 0
        best_move = order[0]
        self.depth_reached = 0
        for depth in range(1, self.max_depth + 1):
            values: dict[Point, float] = {}
            alpha, best_here, best_value = -INF, order[0], -INF
            try:
                for move in order:
                    child = board.copy()
                    child.play(*move)
                    value = -self._negamax(child, depth - 1, -INF, -alpha, deadline)
                    value += 0.01 * static[move]                 # positional tie-break
                    values[move] = value
                    if value > best_value:
                        best_value, best_here = value, move
                    alpha = max(alpha, value)
            except SearchTimeout:
                break
            best_move = best_here
            self.depth_reached = depth
            order.sort(key=lambda m: values[m], reverse=True)
            if best_value >= self.W_CAPTURE * 20:                # won already
                break
        return best_move


INF = float("inf")

KROPKI_AI = {"random": RandomKropki, "heuristic": HeuristicKropki, "search": SearchKropki}

# difficulty -> factory kwargs for SearchKropki (None = heuristic player)
LEVELS = {
    "easy": None,
    "normal": None,
    "hard": {"time_budget": 0.5, "max_depth": 6, "breadth": 6},
    "expert": {"time_budget": 2.0, "max_depth": 8, "breadth": 8},
}


def make_kropki_ai(difficulty: str, seed: Optional[int] = None):
    """easy: heuristic with 35% random moves; normal: heuristic;
    hard: 0.5 s search; expert: 2 s search."""
    if difficulty == "easy":
        return HeuristicKropki(seed, noise=0.35)
    if difficulty == "random":
        return RandomKropki(seed)
    kwargs = LEVELS.get(difficulty)
    if kwargs is None:
        return HeuristicKropki(seed)
    return SearchKropki(seed, **kwargs)
