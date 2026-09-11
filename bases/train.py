"""Training, evaluation and hyperparameter search for the Q-learning agent."""
from __future__ import annotations

import csv
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from .agent import Agent, GreedyAgent, MinimaxAgent, QLearningAgent, RandomAgent
from .config import Config, load_config
from .game import Bases

OPPONENTS: dict[str, type] = {"random": RandomAgent, "greedy": GreedyAgent,
                              "minimax": MinimaxAgent}


@dataclass
class EvalResult:
    opponent: str
    episodes: int
    wins: int
    losses: int
    ties: int
    avg_margin: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes if self.episodes else 0.0

    @property
    def score(self) -> float:
        """Win rate with ties counted as half a win."""
        return (self.wins + 0.5 * self.ties) / self.episodes if self.episodes else 0.0

    def __str__(self) -> str:
        return (f"vs {self.opponent}: {self.wins}W {self.losses}L {self.ties}T "
                f"({self.win_rate:.1%} wins, avg margin {self.avg_margin:+.2f})")


@dataclass
class TrainingStats:
    """A row of the training log; produced at every evaluation point."""
    episode: int
    episodes: int
    exploration_rate: float
    states: int
    entries: int
    elapsed: float
    win_rate_random: Optional[float] = None
    win_rate_greedy: Optional[float] = None
    avg_margin_greedy: Optional[float] = None

    @property
    def fraction(self) -> float:
        return self.episode / self.episodes if self.episodes else 1.0


ProgressCallback = Callable[[TrainingStats], Optional[bool]]


class Observed:
    """Wrap a fixed opponent so its moves also train ``learner``.

    The learner's value function is for "the player to move", so moves made
    by any player are valid off-policy experience for its table.
    """

    def __init__(self, player: Agent, learner: QLearningAgent):
        self.player = player
        self.learner = learner
        self.name = getattr(player, "name", "opponent")

    def choose_index(self, board):
        return self.player.choose_index(board)

    def observe(self, transition):
        self.learner.observe(transition)


def make_opponent(kind: str, seed: Optional[int] = None) -> Agent:
    try:
        return OPPONENTS[kind](seed=seed)
    except KeyError:
        raise ValueError(f"unknown opponent {kind!r}; choose from {sorted(OPPONENTS)}") from None


def evaluate(agent: QLearningAgent, opponent: Agent | str, episodes: int,
             extra_turn_on_box: Optional[bool] = None,
             seed: Optional[int] = None) -> EvalResult:
    """Play ``episodes`` games (alternating sides) without learning."""
    if isinstance(opponent, str):
        opponent = make_opponent(opponent, seed)
    if extra_turn_on_box is None:
        extra_turn_on_box = agent.extra_turn_on_box
    game = Bases(agent.size, extra_turn_on_box=extra_turn_on_box)
    was_training = agent.training_mode
    agent.training_mode = False
    wins = losses = ties = 0
    margin = 0
    try:
        for i in range(episodes):
            side = "A" if i % 2 == 0 else "B"
            result = game.play(agent, opponent) if side == "A" else game.play(opponent, agent)
            margin += game.margin(side)
            if result == side:
                wins += 1
            elif result == "Tie":
                ties += 1
            else:
                losses += 1
    finally:
        agent.training_mode = was_training
    return EvalResult(getattr(opponent, "name", type(opponent).__name__), episodes,
                      wins, losses, ties, margin / episodes if episodes else 0.0)


def exploration_schedule(episode: int, episodes: int, start: float, minimum: float,
                         decay_fraction: float = 0.8) -> float:
    """Exponential decay from ``start`` to ``minimum`` over the first
    ``decay_fraction`` of training, then constant."""
    if episodes <= 0 or start <= minimum:
        return max(minimum, start)
    horizon = max(1.0, decay_fraction * episodes)
    return max(minimum, start * (minimum / start) ** min(1.0, episode / horizon))


def train_agents(size: int, episodes: int, *,
                 learning_rate: Optional[float] = None,
                 discount_factor: Optional[float] = None,
                 exploration_rate: Optional[float] = None,
                 exploration_min: Optional[float] = None,
                 opponent_mix: Optional[dict[str, float]] = None,
                 extra_turn_on_box: Optional[bool] = None,
                 eval_every: Optional[int] = None,
                 eval_episodes: int = 200,
                 progress: Optional[ProgressCallback] = None,
                 seed: Optional[int] = None,
                 save_path: Optional[str | Path] = None,
                 log_path: Optional[str | Path] = None,
                 agent: Optional[QLearningAgent] = None,
                 config: Optional[Config] = None,
                 quiet: bool = False) -> QLearningAgent:
    """Train a Q-learning agent by self-play (plus a mix of fixed opponents).

    Hyperparameters default to the ``training`` section of the config for
    this board size.  ``progress`` is called at each evaluation point with a
    :class:`TrainingStats`; returning ``False`` from it stops training early.
    The policy is saved to ``save_path`` (default: the configured policy path
    for this size; pass ``save_path=False`` to skip saving).
    """
    cfg = config or load_config()
    hp = cfg.hyperparameters(size)
    learning_rate = hp["learning_rate"] if learning_rate is None else learning_rate
    discount_factor = hp["discount_factor"] if discount_factor is None else discount_factor
    exploration_rate = hp["exploration_rate"] if exploration_rate is None else exploration_rate
    exploration_min = hp["exploration_min"] if exploration_min is None else exploration_min
    opponent_mix = dict(hp["opponent_mix"] if opponent_mix is None else opponent_mix)
    if extra_turn_on_box is None:
        extra_turn_on_box = cfg.extra_turn_on_box
    if eval_every is None:
        eval_every = max(1, episodes // 10)

    rng = random.Random(seed)
    if agent is None:
        agent = QLearningAgent(size, learning_rate, discount_factor, exploration_rate,
                               exploration_min, training_mode=True,
                               seed=rng.randrange(1 << 30),
                               extra_turn_on_box=extra_turn_on_box)
    else:
        agent.training_mode = True
        agent.learning_rate, agent.discount_factor = learning_rate, discount_factor
        agent.exploration_rate, agent.exploration_min = exploration_rate, exploration_min
        agent.extra_turn_on_box = extra_turn_on_box

    kinds = [k for k, w in opponent_mix.items() if w > 0]
    weights = [opponent_mix[k] for k in kinds]
    if not kinds:
        kinds, weights = ["self"], [1.0]
    opponents = {k: Observed(make_opponent(k, rng.randrange(1 << 30)), agent)
                 for k in kinds if k != "self"}

    game = Bases(size, extra_turn_on_box=extra_turn_on_box)
    history: list[TrainingStats] = []
    start = time.time()
    eval_seed = rng.randrange(1 << 30)

    def checkpoint(episode: int) -> bool:
        stats = TrainingStats(episode, episodes, agent.exploration_rate,
                              agent.num_states, agent.num_entries, time.time() - start)
        if eval_episodes > 0:
            rnd = evaluate(agent, "random", eval_episodes, extra_turn_on_box, eval_seed)
            grd = evaluate(agent, "greedy", eval_episodes, extra_turn_on_box, eval_seed)
            stats.win_rate_random = rnd.win_rate
            stats.win_rate_greedy = grd.win_rate
            stats.avg_margin_greedy = grd.avg_margin
        history.append(stats)
        if not quiet:
            msg = (f"[{episode:>{len(str(episodes))}}/{episodes}] eps={stats.exploration_rate:.3f} "
                   f"states={stats.states:,} t={stats.elapsed:.0f}s")
            if stats.win_rate_random is not None:
                msg += (f" | win vs random {stats.win_rate_random:.1%}, "
                        f"vs greedy {stats.win_rate_greedy:.1%} "
                        f"(margin {stats.avg_margin_greedy:+.2f})")
            print(msg, flush=True)
        if progress is not None and progress(stats) is False:
            return False
        return True

    stopped = False
    for episode in range(1, episodes + 1):
        agent.exploration_rate = exploration_schedule(episode - 1, episodes,
                                                      exploration_rate, exploration_min)
        kind = rng.choices(kinds, weights)[0]
        if kind == "self":
            game.play(agent, agent)
        elif rng.random() < 0.5:
            game.play(agent, opponents[kind])
        else:
            game.play(opponents[kind], agent)
        agent.episodes_trained += 1
        if episode % eval_every == 0 or episode == episodes:
            if not checkpoint(episode):
                stopped = True
                break
    if stopped and history and history[-1].episode != agent.episodes_trained:
        checkpoint(agent.episodes_trained)

    if save_path is not False:
        path = Path(save_path) if save_path else cfg.path("policy", size)
        agent.save_policy(path)
        if not quiet:
            print(f"Policy saved to {path} ({agent.num_states:,} states, "
                  f"{agent.num_entries:,} entries).")
    if log_path is not False:
        path = Path(log_path) if log_path else cfg.path("training_log", size)
        write_training_log(path, history)
    return agent


def write_training_log(path: Path, history: list[TrainingStats]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        for row in history:
            writer.writerow(asdict(row))


def evaluate_hyperparameters(learning_rate: float, discount_factor: float,
                             exploration_rate: float, grid_size: int,
                             train_episodes: int, test_episodes: int,
                             exploration_min: float = 0.01,
                             extra_turn_on_box: Optional[bool] = None,
                             seed: Optional[int] = None,
                             config: Optional[Config] = None) -> dict[str, float]:
    """Train a fresh agent and return its scores against greedy and random."""
    agent = train_agents(grid_size, train_episodes, learning_rate=learning_rate,
                         discount_factor=discount_factor,
                         exploration_rate=exploration_rate,
                         exploration_min=exploration_min,
                         extra_turn_on_box=extra_turn_on_box, eval_episodes=0,
                         seed=seed, save_path=False, log_path=False,
                         config=config, quiet=True)
    greedy = evaluate(agent, "greedy", test_episodes, extra_turn_on_box, seed)
    rnd = evaluate(agent, "random", test_episodes, extra_turn_on_box, seed)
    return {"score_greedy": greedy.score, "score_random": rnd.score,
            "win_rate_greedy": greedy.win_rate, "win_rate_random": rnd.win_rate,
            "avg_margin_greedy": greedy.avg_margin,
            "score": 0.7 * greedy.score + 0.3 * rnd.score, "states": agent.num_states}


def hyperparameter_tuning(grid_size: int, iterations: int, train_episodes: int,
                          test_episodes: int, *, seed: Optional[int] = None,
                          results_path: Optional[str | Path] = None,
                          extra_turn_on_box: Optional[bool] = None,
                          config: Optional[Config] = None, quiet: bool = False,
                          progress: Optional[Callable[[int, int, dict], Optional[bool]]] = None
                          ) -> dict[str, float]:
    """Random search over learning rate, discount and exploration.

    Every trial trains a new agent for ``train_episodes`` games and scores it
    against the greedy and random opponents (``score`` = 0.7 * greedy +
    0.3 * random, ties counting half).  All trials are written to a CSV.
    """
    cfg = config or load_config()
    rng = random.Random(seed)
    results: list[dict] = []
    best: Optional[dict] = None
    for i in range(1, iterations + 1):
        params = {
            "learning_rate": round(rng.uniform(0.05, 0.9), 4),
            "discount_factor": round(rng.uniform(0.8, 1.0), 4),
            "exploration_rate": round(rng.uniform(0.1, 0.6), 4),
            "exploration_min": round(rng.uniform(0.005, 0.05), 4),
        }
        if not quiet:
            print(f"Trial {i}/{iterations}: " +
                  ", ".join(f"{k}={v}" for k, v in params.items()), flush=True)
        scores = evaluate_hyperparameters(params["learning_rate"], params["discount_factor"],
                                          params["exploration_rate"], grid_size,
                                          train_episodes, test_episodes,
                                          exploration_min=params["exploration_min"],
                                          extra_turn_on_box=extra_turn_on_box,
                                          seed=rng.randrange(1 << 30), config=cfg)
        row = {"trial": i, **params, **scores}
        results.append(row)
        if not quiet:
            print(f"   score={row['score']:.3f} (vs greedy {row['win_rate_greedy']:.1%}, "
                  f"vs random {row['win_rate_random']:.1%})", flush=True)
        if best is None or row["score"] > best["score"]:
            best = row
        if progress is not None and progress(i, iterations, row) is False:
            break

    path = Path(results_path) if results_path else cfg.path("tuning_results", grid_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    if best is not None and not quiet:
        print("\n----- Best hyperparameters -----")
        for key in ("learning_rate", "discount_factor", "exploration_rate", "exploration_min"):
            print(f"{key:>17}: {best[key]}")
        print(f"            score: {best['score']:.3f}")
        print(f"Results written to {path}")
    return best or {}
