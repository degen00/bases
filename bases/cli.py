"""Command-line entry point: ``python play.py <command> [options]``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .agent import MinimaxAgent, QLearningAgent
from .config import load_config
from .game import Bases
from .kropki import KropkiBoard
from .kropki_ai import make_kropki_ai
from .player import HumanPlayer
from .train import evaluate, hyperparameter_tuning, train_agents

DIFFICULTIES = {"easy": (0.35, 0, 0), "normal": (0.0, 0, 0), "hard": (0.0, 3, 2),
                "expert": (0.0, 4, 3)}
KROPKI_LEVELS = ["random", "easy", "normal", "hard", "expert"]
MODES = {
    "hvh": "Human vs Human",
    "hva": "Human (A) vs AI (B)",
    "avh": "AI (A) vs Human (B)",
    "ava": "AI vs AI",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bases",
        description="Bases: Dots and Boxes with a Q-learning opponent. "
                    "Run without a command to open the graphical interface.")
    parser.add_argument("--config", help="path to bases.yml")
    sub = parser.add_subparsers(dest="command")

    def common(p, default_size=3):
        p.add_argument("--size", "-s", type=int, default=default_size,
                       help="boxes per side (default %(default)s)")
        p.add_argument("--extra-turn", action=argparse.BooleanOptionalAction, default=None,
                       help="classic rule: completing a box grants another move "
                            "(default from config)")

    p = sub.add_parser("gui", help="open the graphical interface (default)")
    common(p)
    p.add_argument("--difficulty", choices=list(DIFFICULTIES), default="normal")
    p.add_argument("--game", choices=["boxes", "kropki"], default="boxes")
    p.add_argument("--points", default="10x10",
                   help="Kropki board size as WIDTHxHEIGHT or one number (default %(default)s)")

    p = sub.add_parser("kropki", help="play Kropki (free-form bases) in the terminal")
    p.add_argument("--width", "-W", type=int, default=10)
    p.add_argument("--height", "-H", type=int, default=None)
    p.add_argument("--mode", "-m", choices=list(MODES), default="hva",
                   help="; ".join(f"{k}: {v}" for k, v in MODES.items()))
    p.add_argument("--difficulty", choices=KROPKI_LEVELS, default="normal")
    p.add_argument("--seed", type=int)

    p = sub.add_parser("kropki-eval", help="self-play tournament between Kropki AI levels")
    p.add_argument("--levels", default="random,normal,hard",
                   help="comma-separated subset of " + ",".join(KROPKI_LEVELS))
    p.add_argument("--games", type=int, default=4, help="games per pair (colours alternate)")
    p.add_argument("--width", "-W", type=int, default=8)
    p.add_argument("--height", "-H", type=int, default=None)
    p.add_argument("--seed", type=int)
    p.add_argument("--verbose", "-v", action="store_true")

    p = sub.add_parser("train", help="train the Q-learning agent")
    common(p)
    p.add_argument("--episodes", "-n", type=int, default=20000)
    p.add_argument("--lr", type=float, help="learning rate")
    p.add_argument("--gamma", type=float, help="discount factor")
    p.add_argument("--eps", type=float, help="initial exploration rate")
    p.add_argument("--eps-min", type=float, help="final exploration rate")
    p.add_argument("--eval-every", type=int, help="episodes between evaluations")
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--seed", type=int)
    p.add_argument("--output", "-o", help="policy file to write")
    p.add_argument("--resume", action="store_true",
                   help="continue training the existing policy for this size")

    p = sub.add_parser("tune", help="random search over hyperparameters")
    common(p)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--train-episodes", type=int, default=5000)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--seed", type=int)

    p = sub.add_parser("eval", help="evaluate a trained policy")
    common(p)
    p.add_argument("--episodes", "-n", type=int, default=500)
    p.add_argument("--opponent", choices=["random", "greedy", "minimax", "all"],
                   default="all")
    p.add_argument("--policy", help="policy file (default: configured path)")
    p.add_argument("--depth", type=int, default=0,
                   help="search depth for the agent at play time (0 = table only)")
    p.add_argument("--minimax-depth", type=int, default=2)
    p.add_argument("--seed", type=int)

    p = sub.add_parser("play", help="play in the terminal")
    common(p)
    p.add_argument("--mode", "-m", choices=list(MODES), default=None,
                   help="; ".join(f"{k}: {v}" for k, v in MODES.items()))
    p.add_argument("--games", type=int, default=1, help="number of games (0 = until Ctrl-C)")
    p.add_argument("--policy", help="policy file (default: configured path)")
    p.add_argument("--difficulty", choices=list(DIFFICULTIES), default="normal")
    p.add_argument("--no-log", action="store_true", help="do not append moves to lines.csv")
    return parser


def load_agent(cfg, size: int, policy: Optional[str], extra_turn: bool) -> QLearningAgent:
    path = Path(policy) if policy else cfg.path("policy", size)
    agent = QLearningAgent(size, training_mode=False, extra_turn_on_box=extra_turn)
    if path.is_file():
        agent.load_policy(path)
        print(f"Loaded policy {path} ({agent.num_states:,} states, "
              f"{agent.episodes_trained:,} training games).")
    else:
        print(f"No policy at {path}; the AI will play with the greedy heuristic only. "
              f"Train one with: python play.py train --size {size}")
    return agent


def cmd_train(args, cfg) -> int:
    extra = cfg.extra_turn_on_box if args.extra_turn is None else args.extra_turn
    agent = None
    if args.resume:
        path = Path(args.output) if args.output else cfg.path("policy", args.size)
        if path.is_file():
            agent = QLearningAgent.load(path, training_mode=True)
            print(f"Resuming from {path} ({agent.episodes_trained:,} games so far).")
        else:
            print(f"No policy at {path}; starting from scratch.")
    train_agents(args.size, args.episodes, learning_rate=args.lr,
                 discount_factor=args.gamma, exploration_rate=args.eps,
                 exploration_min=args.eps_min, extra_turn_on_box=extra,
                 eval_every=args.eval_every, eval_episodes=args.eval_episodes,
                 seed=args.seed, save_path=args.output, agent=agent, config=cfg)
    return 0


def cmd_tune(args, cfg) -> int:
    extra = cfg.extra_turn_on_box if args.extra_turn is None else args.extra_turn
    hyperparameter_tuning(args.size, args.trials, args.train_episodes, args.eval_episodes,
                          seed=args.seed, extra_turn_on_box=extra, config=cfg)
    return 0


def cmd_eval(args, cfg) -> int:
    extra = cfg.extra_turn_on_box if args.extra_turn is None else args.extra_turn
    agent = load_agent(cfg, args.size, args.policy, extra)
    agent.search_depth = args.depth
    kinds = ["random", "greedy", "minimax"] if args.opponent == "all" else [args.opponent]
    for kind in kinds:
        opponent = MinimaxAgent(args.minimax_depth, args.seed) if kind == "minimax" else kind
        episodes = args.episodes if kind != "minimax" else max(10, args.episodes // 5)
        print(evaluate(agent, opponent, episodes, extra, args.seed))
    return 0


def cmd_play(args, cfg) -> int:
    extra = cfg.extra_turn_on_box if args.extra_turn is None else args.extra_turn
    mode = args.mode
    if mode is None:
        print("Select game mode:")
        for i, (key, label) in enumerate(MODES.items(), start=1):
            print(f"  {i}. {label} ({key})")
        choice = input("Enter a number or key: ").strip().lower()
        keys = list(MODES)
        mode = keys[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(keys) \
            else choice
        if mode not in MODES:
            print("Invalid choice.")
            return 1
    ai = load_agent(cfg, args.size, args.policy, extra) if "a" in mode else None
    if ai is not None:
        ai.noise, small, large = DIFFICULTIES[args.difficulty]
        ai.search_depth = small if args.size <= 3 else large
    players = {"h": HumanPlayer, "a": lambda: ai}
    player_a = players[mode[0]]()
    player_b = players[mode[2]]()

    logging.basicConfig(filename=cfg.path("game_log"), level=logging.INFO,
                        format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    game = Bases(args.size, extra_turn_on_box=extra, verbose=True,
                 log_moves=not args.no_log, lines_log_path=cfg.path("lines_log"))
    print(f"\n{MODES[mode]} on a {args.size}x{args.size} board. "
          f"Dots are (x, y) with (0, 0) top-left; enter a move as 'x1 y1 x2 y2'.")
    played = 0
    try:
        while args.games == 0 or played < args.games:
            game.play(player_a, player_b)
            played += 1
    except (KeyboardInterrupt, EOFError):
        print("\nGame interrupted.")
    return 0


def cmd_gui(args, cfg) -> int:
    try:
        from .gui import run
    except ImportError as exc:
        print(f"The GUI needs pygame: pip install pygame ({exc})")
        return 1
    extra = cfg.extra_turn_on_box if args.extra_turn is None else args.extra_turn
    run(size=args.size, extra_turn_on_box=extra, config=cfg, difficulty=args.difficulty,
        game=args.game, kropki_size=parse_points(args.points))
    return 0


def parse_points(text: str) -> tuple[int, int]:
    """'39x32' -> (39, 32); '10' -> (10, 10)."""
    parts = str(text).lower().replace("*", "x").split("x")
    try:
        w = int(parts[0])
        h = int(parts[1]) if len(parts) > 1 and parts[1] else w
    except ValueError:
        raise SystemExit(f"invalid board size {text!r}; use WIDTHxHEIGHT, e.g. 39x32") from None
    return w, h


def cmd_kropki_eval(args, cfg) -> int:
    from .kropki_tournament import tournament
    levels = [lvl.strip() for lvl in args.levels.split(",") if lvl.strip()]
    unknown = [lvl for lvl in levels if lvl not in KROPKI_LEVELS]
    if unknown or len(levels) < 2:
        print(f"choose at least two of {', '.join(KROPKI_LEVELS)}")
        return 1
    height = args.height or args.width
    print(f"Kropki {args.width}x{height}, {args.games} games per pair, colours alternate.")
    tournament(levels, args.games, args.width, height, args.seed, verbose=args.verbose)
    return 0


def cmd_kropki(args, cfg) -> int:
    board = KropkiBoard(args.width, args.height)
    ai = {p: make_kropki_ai(args.difficulty, args.seed)
          for p, c in zip("AB", (args.mode[0], args.mode[2])) if c == "a"}
    print(f"\nKropki {board.width}x{board.height}: {MODES[args.mode]}. Enter a point as "
          "'x y' ((0, 0) is top-left), or 'pass', 'undo', 'quit'.")
    try:
        while not board.is_over():
            print()
            print(board.render())
            if board.turn in ai:
                move = ai[board.turn].choose_move(board)
                if move is None:
                    board.pass_turn()
                    print(f"AI (Player {board.turn}) passes")
                else:
                    mine, theirs = board.play(*move)
                    print(f"AI plays {move}" + (f", capturing {mine}" if mine else ""))
                continue
            raw = input(f"Player {board.turn}: ").strip().lower()
            if raw == "quit":
                break
            if raw == "pass":
                board.pass_turn()
                continue
            if raw == "undo":
                board.undo()
                while board.history and board.turn in ai:
                    board.undo()
                continue
            parts = raw.split()
            try:
                x, y = int(parts[0]), int(parts[1])
                mine, theirs = board.play(x, y)
            except (ValueError, IndexError) as exc:
                print(f"Invalid move: {exc if str(exc) else 'enter two integers'}")
                continue
            if mine:
                print(f"You captured {mine} dot(s)!")
            if theirs:
                print(f"Your dot was captured ({theirs}).")
    except (KeyboardInterrupt, EOFError):
        print("\nGame interrupted.")
        return 0
    print()
    print(board.render())
    if board.is_over():
        print(f"Game over: {board.winner()}  (A {board.scores['A']} - B {board.scores['B']})")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    raw = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(raw)
    if args.command is None:
        args = parser.parse_args(raw + ["gui"])
    cfg = load_config(args.config)
    commands = {"gui": cmd_gui, "train": cmd_train, "tune": cmd_tune,
                "eval": cmd_eval, "play": cmd_play, "kropki": cmd_kropki,
                "kropki-eval": cmd_kropki_eval}
    return commands[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
