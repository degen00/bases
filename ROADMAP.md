# Roadmap

The Python package is the reference implementation and the AI lab: rules are
specified here, AI ideas are tested here with self-play, and other ports must
match this engine exactly. Items are ordered by priority. Status: `todo`,
`in progress`, `done`.

## 1. Conformance vectors — `done`

Recorded games (moves, captures per move, final dots, territory and scores)
exported as JSON so that any port of the Kropki engine can prove it behaves
identically. `python tools/export_vectors.py` regenerates
`data/vectors/kropki_vectors.json`; `tests/test_vectors.py` replays the
committed file against the engine.

## 2. Kropki AI: time-budgeted search — `done`

Replaced the fixed 2-ply search with iterative deepening alpha-beta over a
heuristic beam, a per-move time budget, and a transposition table. Adds an
*Expert* level. The key enabler was `capture_map`: an articulation-point
analysis that lists every capturing move, threat and house point exactly in
one O(cells) pass, so the search no longer needs trial plays for ordering
or leaf evaluation (Expert went from depth 3 / 160 nodes to depth 6 /
2,800 nodes in 2 s). Lesson recorded: a liberty-only approximation made the
deeper search *weaker* than the one-ply exact heuristic until this landed.

## 3. Self-play tournament harness — `done`

`python play.py kropki-eval` plays seeded matches between AI levels (and
custom weight sets), reporting win rates, average margins and ms per move.
Used to tune heuristic weights and to confirm Expert > Hard > Normal > Random.

## 4. Continuous integration — `done`

GitHub Actions (`.github/workflows/tests.yml`) runs the unit tests headlessly
on Python 3.10 and 3.12 and replays the conformance vectors on every push to
`main`/`develop` and on every pull request.

## 5. Classic board sizes — `done`

Kropki boards need not be square; the GUI offers the classic 39x32 (and
25x25) and scales cells down to fit.

## 6. Monte-Carlo tree search for Kropki — `todo`

MCTS with the heuristic as playout policy. Playouts cost ~30 ms each in
Python, which is too slow to matter within a 1 s budget; this item is
intended for the Dart/mobile port, with the Python version as a correctness
reference on tiny boards.

## 7. Kropki GUI zoom and pan — `todo`

Mouse-wheel zoom and drag-to-pan for 39x32 boards; a magnifier or
confirm-move mode for dense grids (also the design the mobile app needs).

## 8. Dots and Boxes beyond the table — `todo`

The tabular Q-learner plateaus (~70% vs greedy on 3x3, prior-only on 4x4+).
Options: linear function approximation over chain/box features, or a small
network trained by self-play. Keep the exact 2x2 solver as ground truth.

## 9. Housekeeping — `todo`

Keep `develop` in sync with `main` after each merge; tag releases
(`v0.2`, `v0.3`, ...); publish policy files as release assets instead of
committing large binaries.
