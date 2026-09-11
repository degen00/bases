import unittest

from bases.board import Board, Geometry


class GeometryTests(unittest.TestCase):
    def test_line_counts(self):
        for n in (1, 2, 3, 5):
            g = Geometry.get(n)
            self.assertEqual(g.num_lines, 2 * n * (n + 1))
            self.assertEqual(len(g.coords), g.num_lines)
            self.assertEqual(len(set(g.coords)), g.num_lines)

    def test_index_roundtrip_and_normalisation(self):
        g = Geometry.get(3)
        for i, move in enumerate(g.coords):
            self.assertEqual(g.index_of(move), i)
            x1, y1, x2, y2 = move
            self.assertEqual(g.index_of((x2, y2, x1, y1)), i)

    def test_invalid_moves(self):
        g = Geometry.get(2)
        for move in [(0, 0, 0, 0), (0, 0, 1, 1), (0, 0, 2, 0), (3, 0, 4, 0), (-1, 0, 0, 0)]:
            with self.assertRaises(ValueError):
                g.index_of(move)

    def test_every_box_has_four_distinct_sides(self):
        g = Geometry.get(3)
        for sides in g.box_sides:
            self.assertEqual(len(set(sides)), 4)
        for idx, boxes in enumerate(g.line_boxes):
            self.assertIn(len(boxes), (1, 2))
            for b in boxes:
                self.assertIn(idx, g.box_sides[b])


class BoardTests(unittest.TestCase):
    def test_capture_and_scores(self):
        b = Board(2)
        b.make_move(0, 0, 1, 0)  # A
        b.make_move(0, 0, 0, 1)  # B
        b.make_move(1, 0, 1, 1)  # A
        self.assertEqual(b.turn, "B")
        self.assertEqual(b.captures_for(b.geom.index_of((0, 1, 1, 1))), 1)
        captured = b.make_move(0, 1, 1, 1)  # B completes the box
        self.assertEqual(captured, 1)
        self.assertEqual(b.scores, {"A": 0, "B": 1})
        self.assertEqual(b.boxes[0][0], "B")
        self.assertEqual(b.turn, "A", "Bases rule: turn alternates even after a box")

    def test_extra_turn_rule(self):
        b = Board(2, extra_turn_on_box=True)
        b.make_move(0, 0, 1, 0)
        b.make_move(0, 0, 0, 1)
        b.make_move(1, 0, 1, 1)
        b.make_move(0, 1, 1, 1)
        self.assertEqual(b.turn, "B", "classic rule: B moves again after a box")

    def test_illegal_moves_raise(self):
        b = Board(2)
        b.make_move(0, 0, 1, 0)
        with self.assertRaises(ValueError):
            b.make_move(1, 0, 0, 0)
        with self.assertRaises(ValueError):
            b.make_move(0, 0, 1, 1)
        self.assertEqual(len(b.history), 1)

    def test_undo_restores_everything(self):
        b = Board(2)
        moves = [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1)]
        snapshots = []
        for m in moves:
            snapshots.append((b.mask, list(b.owner), dict(b.scores), b.turn))
            b.make_move(*m)
        for snap in reversed(snapshots):
            self.assertTrue(b.undo())
            self.assertEqual((b.mask, b.owner, b.scores, b.turn), snap)
        self.assertFalse(b.undo())

    def test_full_game_and_winner(self):
        b = Board(1)
        for move in b.geom.coords:
            b.make_move(*move)
        self.assertTrue(b.is_full())
        self.assertEqual(b.scores, {"A": 0, "B": 1})
        self.assertEqual(b.winner(), "B")
        self.assertEqual(b.margin("B"), 1)

    def test_available_moves_shrink(self):
        b = Board(3)
        self.assertEqual(len(b.available_moves()), 24)
        b.make_move(0, 0, 1, 0)
        self.assertEqual(len(b.available()), 23)
        self.assertNotIn((0, 0, 1, 0), b.available_moves())

    def test_gifts(self):
        b = Board(1)
        b.make_move(0, 0, 1, 0)
        b.make_move(0, 0, 0, 1)
        idx = b.geom.index_of((1, 0, 1, 1))
        self.assertEqual(b.gifts_for(idx), 1)
        self.assertEqual(b.captures_for(idx), 0)

    def test_render_matches_original_layout(self):
        b = Board(2)
        b.make_move(0, 0, 1, 0)
        expected = "\n".join([
            "---------",
            "*---*   *",
            "         ",
            "*   *   *",
            "         ",
            "*   *   *",
            "---------",
        ])
        self.assertEqual(b.render(), expected)


if __name__ == "__main__":
    unittest.main()
