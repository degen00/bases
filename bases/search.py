"""Depth-limited negamax search with alpha-beta pruning.

Values are box margins for the player to move, in the same units as the
Q-table, so a Q-learning agent can use its learned values as the leaf
evaluation and a plain minimax agent can use the heuristic prior.
"""
from __future__ import annotations

from typing import Callable

from .board import Board

INF = float("inf")
LeafValue = Callable[[Board], float]


def prior_leaf(board: Board) -> float:
    """Leaf value from the heuristic prior: best immediate capture/gift balance."""
    available = board.available()
    if not available:
        return 0.0
    prior = board.geom.prior_row(board.mask)
    return max(prior[a] for a in available)


def negamax(board: Board, depth: int, leaf: LeafValue,
            alpha: float = -INF, beta: float = INF) -> float:
    """Best margin the player to move can secure within ``depth`` plies."""
    if board.is_full():
        return 0.0
    if depth == 0:
        return leaf(board)
    available = board.available()
    prior = board.geom.prior_row(board.mask)
    available.sort(key=prior.__getitem__, reverse=True)   # captures first
    best = -INF
    mover = board.turn
    for a in available:
        captured = board.play_index(a)
        if board.turn == mover:
            value = captured + negamax(board, depth - 1, leaf,
                                       alpha - captured, beta - captured)
        else:
            value = captured - negamax(board, depth - 1, leaf,
                                       captured - beta, captured - alpha)
        board.undo()
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def search_values(board: Board, depth: int, leaf: LeafValue) -> dict[int, float]:
    """Value of every available line after a ``depth``-ply search."""
    values = {}
    mover = board.turn
    for a in board.available():
        captured = board.play_index(a)
        if board.turn == mover:
            values[a] = captured + negamax(board, depth - 1, leaf)
        else:
            values[a] = captured - negamax(board, depth - 1, leaf)
        board.undo()
    return values
