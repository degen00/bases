"""Players that are not human: random, greedy and the tabular Q-learner."""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any, Optional

from .board import Board, Geometry, Move, Transition
from .search import prior_leaf, search_values

POLICY_FORMAT = 2


class Agent:
    """Interface every computer player implements."""

    name = "agent"

    def choose_index(self, board: Board) -> int:
        raise NotImplementedError

    def choose_action(self, board: Board) -> Move:
        return board.geom.coords[self.choose_index(board)]


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_index(self, board: Board) -> int:
        return self.rng.choice(board.available())


def heuristic_score(board: Board, idx: int) -> tuple[int, int]:
    """Higher is better: capture boxes, avoid handing 3-sided boxes over."""
    return (board.captures_for(idx), -board.gifts_for(idx))


class GreedyAgent(Agent):
    """Takes any box it can, otherwise avoids giving a third side away."""

    name = "greedy"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_index(self, board: Board) -> int:
        available = board.available()
        best = max(heuristic_score(board, i) for i in available)
        return self.rng.choice([i for i in available if heuristic_score(board, i) == best])


class MinimaxAgent(Agent):
    """Fixed-depth negamax search with the heuristic prior at the leaves.

    A stronger benchmark than :class:`GreedyAgent` (depth 1 is greedy with a
    one-ply reply check; depth 2-3 sees simple traps).
    """

    name = "minimax"

    def __init__(self, depth: int = 2, seed: Optional[int] = None):
        self.depth = max(1, depth)
        self.rng = random.Random(seed)

    def choose_index(self, board: Board) -> int:
        values = search_values(board, self.depth, prior_leaf)
        best = max(values.values())
        return self.rng.choice([a for a, v in values.items() if v == best])


class QLearningAgent(Agent):
    """Tabular Q-learning for a two-player, alternating-move game.

    * State = canonical (symmetry-reduced) bitmask of drawn lines.  Box
      ownership is not needed: the value of a position *for the player to
      move* depends only on the lines.
    * Reward = boxes captured by the move.  Summed over a game with the
      negamax target below this is exactly the final box margin, so no
      separate win/loss reward is required.
    * Target = ``r + gamma * V(s')`` where ``V(s') = max_a Q(s', a)`` if the
      same player moves again (extra-turn rule) and ``-max_a Q(s', a)`` when
      the opponent moves next.  Both sides of a self-play game can therefore
      share one table.
    * Unseen (state, action) pairs start from a heuristic prior (boxes the
      line completes minus boxes it gives away, see ``Geometry.prior_row``)
      instead of 0, so an untrained agent already plays like the greedy
      heuristic and learning refines it.  Pass ``use_prior=False`` for the
      plain zero initialisation.
    * At play time ``search_depth`` > 0 runs a negamax search of that many
      plies using the table (plus prior) as leaf evaluation, and ``noise`` is
      the probability of a random move (for easier difficulty levels).
    * Updates are applied at the end of each episode in *reverse* order
      (``backward_replay=True``), so the final outcome propagates through the
      whole game in one pass instead of one step per game.
    """

    name = "q-learning"

    def __init__(self, size: int, learning_rate: float = 0.3,
                 discount_factor: float = 0.98, exploration_rate: float = 0.3,
                 exploration_min: float = 0.02, training_mode: bool = True,
                 use_symmetry: bool = True, seed: Optional[int] = None,
                 extra_turn_on_box: bool = False, backward_replay: bool = True,
                 use_prior: bool = True, search_depth: int = 0, noise: float = 0.0):
        self.size = size
        self.geom = Geometry.get(size)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.training_mode = training_mode
        self.use_symmetry = use_symmetry
        self.extra_turn_on_box = extra_turn_on_box
        self.rng = random.Random(seed)
        self.backward_replay = backward_replay
        self.use_prior = use_prior
        self.search_depth = search_depth
        self.noise = noise
        self.q: dict[int, dict[int, float]] = {}
        self.episodes_trained = 0
        self._episode: list[Transition] = []

    # -- table access -----------------------------------------------------
    @property
    def q_table(self) -> dict[int, dict[int, float]]:
        return self.q

    @property
    def num_states(self) -> int:
        return len(self.q)

    @property
    def num_entries(self) -> int:
        return sum(len(row) for row in self.q.values())

    _IDENTITY = (0,)

    def _canon(self, mask: int) -> tuple[int, tuple[int, ...]]:
        return self.geom.canonical(mask) if self.use_symmetry else (mask, self._IDENTITY)

    def _canon_action(self, ks: tuple[int, ...], action: int) -> int:
        return self.geom.canonical_action(ks, action)

    _ZERO_ROW: tuple[float, ...] = ()

    def _prior(self, cmask: int) -> tuple[float, ...]:
        if self.use_prior:
            return self.geom.prior_row(cmask)
        if len(self._ZERO_ROW) != self.geom.num_lines:
            self._ZERO_ROW = (0.0,) * self.geom.num_lines
        return self._ZERO_ROW

    def _values(self, mask: int, available) -> list[tuple[float, int]]:
        """``[(q, line)]`` for the given lines, in the board's own frame."""
        cmask, ks = self._canon(mask)
        prior = self._prior(cmask)
        row = self.q.get(cmask)
        if len(ks) == 1:
            perm = self.geom.perms[ks[0]]
            if row is None:
                return [(prior[perm[a]], a) for a in available]
            return [(row.get(perm[a], prior[perm[a]]), a) for a in available]
        ca = self._canon_action
        if row is None:
            return [(prior[ca(ks, a)], a) for a in available]
        return [(row.get(ca(ks, a), prior[ca(ks, a)]), a) for a in available]

    def q_values(self, board: Board) -> dict[int, float]:
        """Q-value of every available line, keyed by line index."""
        return {a: q for q, a in self._values(board.mask, board.available())}

    def knows(self, board: Board) -> bool:
        """Whether the position has been updated at least once."""
        return self._canon(board.mask)[0] in self.q

    def state_value(self, mask: int, available) -> float:
        """``max_a Q(mask, a)`` over the given available lines."""
        if not available:
            return 0.0
        return max(q for q, _ in self._values(mask, available))

    def leaf_value(self, board: Board) -> float:
        """Learned value of ``board`` for the player to move (search leaf)."""
        return self.state_value(board.mask, board.available())

    def action_values(self, board: Board, depth: Optional[int] = None) -> dict[int, float]:
        """Value of every available line: the Q-table when ``depth`` is 0,
        otherwise a ``depth``-ply search with the table as leaf evaluation."""
        depth = self.search_depth if depth is None else depth
        if depth > 0:
            return search_values(board, depth, self.leaf_value)
        return self.q_values(board)

    # -- acting -----------------------------------------------------------
    def choose_index(self, board: Board) -> int:
        available = board.available()
        if self.training_mode:
            if self.rng.random() < self.exploration_rate:
                return self.rng.choice(available)
            values = self._values(board.mask, available)
            best_q = max(q for q, _ in values)
            return self.rng.choice([a for q, a in values if q == best_q])
        if self.noise > 0 and self.rng.random() < self.noise:
            return self.rng.choice(available)
        values = self.action_values(board)
        best_q = max(values.values())
        best = [a for a, q in values.items() if q == best_q]
        if len(best) == 1:
            return best[0]
        return self._greedy(board, best)

    def _greedy(self, board: Board, candidates: list[int]) -> int:
        best = max(heuristic_score(board, i) for i in candidates)
        return self.rng.choice([i for i in candidates
                                if heuristic_score(board, i) == best])

    # -- learning ---------------------------------------------------------
    def observe(self, t: Transition) -> None:
        """Learn from one transition (called by the game loop for every move).

        With ``backward_replay`` the transition is buffered and the whole
        episode is applied newest-first once the terminal move arrives.
        """
        if not self.training_mode:
            return
        if not self.backward_replay:
            self._apply(t)
            return
        self._episode.append(t)
        if t.terminal:
            self.flush()

    def flush(self) -> None:
        """Apply buffered transitions (newest first) and clear the buffer."""
        for t in reversed(self._episode):
            self._apply(t)
        self._episode.clear()

    def _apply(self, t: Transition) -> None:
        cs, ks = self._canon(t.state)
        ca = self._canon_action(ks, t.action)
        if t.terminal:
            future = 0.0
        else:
            future = self.state_value(t.next_state, t.next_available)
            if not t.same_mover:
                future = -future
        target = t.reward + self.discount_factor * future
        row = self.q.setdefault(cs, {})
        old = row.get(ca)
        if old is None:
            old = self._prior(cs)[ca]
        row[ca] = old + self.learning_rate * (target - old)

    def update(self, previous_state: int, action: int, reward: float,
               next_state: int, available_moves, same_mover: bool = False,
               terminal: bool = False) -> None:
        """Explicit-argument form of a single, immediate Q-update."""
        if self.training_mode:
            self._apply(Transition(previous_state, action, reward, next_state,
                                   tuple(available_moves), same_mover, terminal, ""))

    # -- persistence ------------------------------------------------------
    def metadata(self) -> dict[str, Any]:
        return {
            "format": POLICY_FORMAT,
            "size": self.size,
            "extra_turn_on_box": self.extra_turn_on_box,
            "use_symmetry": self.use_symmetry,
            "use_prior": self.use_prior,
            "episodes_trained": self.episodes_trained,
            "states": self.num_states,
            "entries": self.num_entries,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate,
                "exploration_min": self.exploration_min,
            },
        }

    @staticmethod
    def _open(path: Path, mode: str):
        """Open a policy file; ``.gz`` paths are gzip-compressed JSON."""
        if path.suffix == ".gz":
            return gzip.open(path, mode + "t", encoding="utf-8")
        return open(path, mode, encoding="utf-8")

    def save_policy(self, file_path) -> Path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.metadata()
        payload["q"] = {str(s): {str(a): round(q, 5) for a, q in row.items()}
                        for s, row in self.q.items()}
        with self._open(path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        return path

    @classmethod
    def read_metadata(cls, file_path) -> dict[str, Any]:
        """Read a policy file's metadata (everything but the table)."""
        with cls._open(Path(file_path), "r") as f:
            payload = json.load(f)
        payload.pop("q", None)
        return payload

    def load_policy(self, file_path) -> None:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No trained policy at {path}. Train the agent first.")
        with self._open(path, "r") as f:
            payload = json.load(f)
        if payload.get("format") != POLICY_FORMAT:
            raise ValueError(f"{path} is not a format-{POLICY_FORMAT} policy file; "
                             "please retrain.")
        if payload.get("size") != self.size:
            raise ValueError(f"{path} was trained for a {payload.get('size')}x"
                             f"{payload.get('size')} board, not {self.size}x{self.size}.")
        self.use_symmetry = bool(payload.get("use_symmetry", True))
        self.use_prior = bool(payload.get("use_prior", True))
        self.extra_turn_on_box = bool(payload.get("extra_turn_on_box", False))
        self.episodes_trained = int(payload.get("episodes_trained", 0))
        hp = payload.get("hyperparameters", {})
        for key in ("learning_rate", "discount_factor", "exploration_rate", "exploration_min"):
            if key in hp:
                setattr(self, key, float(hp[key]))
        self.q = {int(s): {int(a): float(q) for a, q in row.items()}
                  for s, row in payload["q"].items()}

    @classmethod
    def load(cls, file_path, training_mode: bool = False, **kwargs) -> "QLearningAgent":
        meta = cls.read_metadata(file_path)
        agent = cls(int(meta["size"]), training_mode=training_mode, **kwargs)
        agent.load_policy(file_path)
        return agent
