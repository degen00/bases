#!/usr/bin/env python3
"""Export and verify Kropki conformance vectors.

A vector is one recorded game: board size, rule flags, every move with the
captures it produced and the scores after it, and the final dots/territory.
A port of the engine (for example the Dart one for the mobile app) replays
the moves and must reproduce every field.  ``verify`` does the same replay
against this engine, which is what ``tests/test_vectors.py`` runs.

    python tools/export_vectors.py                 # writes data/vectors/kropki_vectors.json
    python tools/export_vectors.py --games 100 --seed 3 -o /tmp/v.json
    python tools/export_vectors.py --verify data/vectors/kropki_vectors.json

Format (version 1)::

    {"format": 1, "game": "kropki", "games": [
       {"id": 0, "width": 6, "height": 6, "mover_priority": true, "allow_pass": true,
        "players": {"A": "random", "B": "heuristic"},
        "moves": [{"player": "A", "x": 2, "y": 3, "captured": 0, "lost": 0,
                   "scores": [0, 0]},                     # or {"player": "B", "pass": true, ...}
                  ...],
        "final": {"scores": [3, 1], "winner": "A", "over": true,
                  "dots": ["..A.B.", ...],               # A/B/. per point, row by row
                  "territory": ["......", ...]}}]}       # A/B/. territory owner
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bases.kropki import KropkiBoard  # noqa: E402
from bases.kropki_ai import HeuristicKropki, RandomKropki  # noqa: E402

FORMAT = 1
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "vectors" / "kropki_vectors.json"

# (width, height, players A/B, mover_priority) cycled through the games
SCENARIOS = [
    (5, 5, ("random", "random"), True),
    (6, 6, ("random", "heuristic"), True),
    (6, 6, ("heuristic", "random"), False),
    (7, 5, ("random", "random"), True),
    (8, 8, ("heuristic", "heuristic"), True),
    (8, 8, ("random", "random"), False),
    (10, 10, ("heuristic", "random"), True),
    (4, 4, ("random", "random"), True),
]


def make_player(kind: str, seed: int):
    return HeuristicKropki(seed) if kind == "heuristic" else RandomKropki(seed)


def _rows(grid) -> list[str]:
    return ["".join(cell or "." for cell in row) for row in grid]


def record_game(game_id: int, width: int, height: int, players: tuple[str, str],
                mover_priority: bool, seed: int, pass_chance: float = 0.02) -> dict:
    rng = random.Random(seed)
    board = KropkiBoard(width, height, mover_priority=mover_priority, allow_pass=True)
    agents = {"A": make_player(players[0], rng.randrange(1 << 30)),
              "B": make_player(players[1], rng.randrange(1 << 30))}
    moves = []
    while not board.is_over():
        player = board.turn
        if rng.random() < pass_chance:
            board.pass_turn()
            moves.append({"player": player, "pass": True, "captured": 0, "lost": 0,
                          "scores": [board.scores["A"], board.scores["B"]]})
            continue
        move = agents[player].choose_move(board)
        if move is None:
            board.pass_turn()
            moves.append({"player": player, "pass": True, "captured": 0, "lost": 0,
                          "scores": [board.scores["A"], board.scores["B"]]})
            continue
        mine, theirs = board.play(*move)
        moves.append({"player": player, "x": move[0], "y": move[1],
                      "captured": mine, "lost": theirs,
                      "scores": [board.scores["A"], board.scores["B"]]})
    return {
        "id": game_id, "width": width, "height": height,
        "mover_priority": mover_priority, "allow_pass": True,
        "players": {"A": players[0], "B": players[1]},
        "moves": moves,
        "final": {"scores": [board.scores["A"], board.scores["B"]],
                  "winner": board.winner(), "over": board.is_over(),
                  "dots": _rows(board.dot), "territory": _rows(board.territory)},
    }


def export(games: int, seed: int) -> dict:
    rng = random.Random(seed)
    records = []
    for i in range(games):
        width, height, players, priority = SCENARIOS[i % len(SCENARIOS)]
        records.append(record_game(i, width, height, players, priority, rng.randrange(1 << 30)))
    return {"format": FORMAT, "game": "kropki", "seed": seed, "games": records}


def verify(record: dict) -> list[str]:
    """Replay one recorded game; return a list of mismatch descriptions."""
    problems: list[str] = []
    board = KropkiBoard(record["width"], record["height"],
                        mover_priority=record["mover_priority"],
                        allow_pass=record.get("allow_pass", True))
    for i, move in enumerate(record["moves"]):
        if board.turn != move["player"]:
            problems.append(f"move {i}: expected player {move['player']}, engine says {board.turn}")
            break
        if move.get("pass"):
            board.pass_turn()
            got = (0, 0)
        else:
            try:
                got = board.play(move["x"], move["y"])
            except ValueError as exc:
                problems.append(f"move {i}: engine rejected ({move['x']}, {move['y']}): {exc}")
                break
        if got != (move["captured"], move["lost"]):
            problems.append(f"move {i}: captures {got} != {(move['captured'], move['lost'])}")
        scores = [board.scores["A"], board.scores["B"]]
        if scores != move["scores"]:
            problems.append(f"move {i}: scores {scores} != {move['scores']}")
    final = record["final"]
    if [board.scores["A"], board.scores["B"]] != final["scores"]:
        problems.append(f"final scores {board.scores} != {final['scores']}")
    if board.winner() != final["winner"]:
        problems.append(f"winner {board.winner()} != {final['winner']}")
    if board.is_over() != final["over"]:
        problems.append(f"over {board.is_over()} != {final['over']}")
    if _rows(board.dot) != final["dots"]:
        problems.append("final dots differ")
    if _rows(board.territory) != final["territory"]:
        problems.append("final territory differ")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--verify", metavar="FILE", help="replay FILE instead of exporting")
    args = parser.parse_args(argv)

    if args.verify:
        with open(args.verify, "r", encoding="utf-8") as f:
            data = json.load(f)
        bad = 0
        for record in data["games"]:
            problems = verify(record)
            if problems:
                bad += 1
                print(f"game {record['id']}: " + "; ".join(problems[:3]))
        print(f"{len(data['games']) - bad}/{len(data['games'])} games replay identically")
        return 1 if bad else 0

    data = export(args.games, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    moves = sum(len(g["moves"]) for g in data["games"])
    captures = sum(m["captured"] + m["lost"] for g in data["games"] for m in g["moves"])
    print(f"Wrote {len(data['games'])} games, {moves} moves, {captures} captured dots to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
