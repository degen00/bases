"""Exact negamax solver for tiny boards, used as ground truth in tests."""
from functools import lru_cache

from bases.board import Board, Geometry


class OptimalAgent:
    """Plays perfectly (only feasible for 1x1 and 2x2 boards)."""

    name = "optimal"

    def __init__(self, size: int, extra_turn_on_box: bool = False):
        self.geom = Geometry.get(size)
        self.extra_turn_on_box = extra_turn_on_box
        self.value = lru_cache(maxsize=None)(self._value)

    def _value(self, mask: int) -> int:
        """Box margin the player to move achieves with optimal play."""
        g = self.geom
        if mask == g.full_mask:
            return 0
        best = -g.num_boxes
        for i in range(g.num_lines):
            if mask >> i & 1:
                continue
            nm = mask | 1 << i
            cap = sum(1 for b in g.line_boxes[i] if nm & g.box_masks[b] == g.box_masks[b])
            v = cap + (self.value(nm) if cap and self.extra_turn_on_box else -self.value(nm))
            best = max(best, v)
        return best

    def action_values(self, board: Board) -> dict[int, int]:
        out = {}
        for i in board.available():
            nm = board.mask | 1 << i
            cap = board.captures_for(i)
            out[i] = cap + (self.value(nm) if cap and board.extra_turn_on_box
                            else -self.value(nm))
        return out

    def choose_index(self, board: Board) -> int:
        values = self.action_values(board)
        return max(values, key=values.get)
