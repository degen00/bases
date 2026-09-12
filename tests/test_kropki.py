import unittest

from bases.kropki import KropkiBoard


def play_all(board, moves):
    """Play alternating moves given as [(A move), (B move), ...]."""
    for move in moves:
        if move is None:
            board.pass_turn()
        else:
            board.play(*move)


class KropkiRulesTests(unittest.TestCase):
    def diamond_capture(self):
        """A surrounds B's dot at (2,2) with a diamond of four dots."""
        b = KropkiBoard(6)
        # A: (2,1) B: (2,2) A: (1,2) B: (0,0) A: (3,2) B: (0,1) A: (2,3) -> capture
        play_all(b, [(2, 1), (2, 2), (1, 2), (0, 0), (3, 2), (0, 1)])
        return b

    def test_diagonal_chain_encloses_enemy_dot(self):
        b = self.diamond_capture()
        self.assertEqual(b.scores, {"A": 0, "B": 0})
        mine, theirs = b.play(2, 3)
        self.assertEqual((mine, theirs), (1, 0))
        self.assertEqual(b.scores, {"A": 1, "B": 0})
        self.assertEqual(b.territory[2][2], "A")
        self.assertFalse(b.is_live(2, 2, "B"), "captured dot is dead")
        self.assertEqual(len(b.bases), 1)
        self.assertEqual(b.turn, "B")

    def test_empty_enclosure_is_a_house_not_a_base(self):
        b = KropkiBoard(6)
        play_all(b, [(2, 1), (0, 0), (1, 2), (0, 1), (3, 2), (0, 2), (2, 3)])
        self.assertEqual(b.scores["A"], 0)
        self.assertIsNone(b.territory[2][2])
        self.assertTrue(b.is_playable(2, 2))
        self.assertEqual(b.bases, [])

    def test_playing_into_a_house_is_captured(self):
        b = KropkiBoard(6)
        play_all(b, [(2, 1), (0, 0), (1, 2), (0, 1), (3, 2), (0, 2), (2, 3)])
        mine, theirs = b.play(2, 2)          # B plays inside A's house
        self.assertEqual((mine, theirs), (0, 1))
        self.assertEqual(b.scores, {"A": 1, "B": 0})
        self.assertEqual(b.territory[2][2], "A")

    def test_edge_does_not_count_as_wall(self):
        b = KropkiBoard(6)
        # B at the corner (0,0); A on (1,0) and (0,1) does NOT enclose it.
        play_all(b, [(1, 0), (0, 0), (0, 1), (5, 5), (1, 1)])
        self.assertEqual(b.scores["A"], 0)
        self.assertTrue(b.is_live(0, 0, "B"))

    def mutual_enclosure(self, mover_priority):
        """A's dots (4,2),(3,3),(5,3) and B's dots (4,3),(3,4),(5,4),(4,5) are
        set up so that A playing (4,4) closes A's diamond around B's (4,3)
        while B's diamond closes around A's new dot at the same time."""
        b = KropkiBoard(9, mover_priority=mover_priority)
        play_all(b, [(4, 2), (4, 3), (3, 3), (3, 4), (5, 3), (5, 4), (0, 0), (4, 5)])
        return b

    def test_mover_priority(self):
        b = self.mutual_enclosure(True)
        mine, theirs = b.play(4, 4)
        self.assertEqual((mine, theirs), (1, 0))
        self.assertEqual(b.territory[3][4], "A")
        self.assertTrue(b.is_live(4, 4, "A"))

    def test_opponent_priority_variant(self):
        b = self.mutual_enclosure(False)
        mine, theirs = b.play(4, 4)
        self.assertEqual((mine, theirs), (0, 1))
        self.assertEqual(b.territory[4][4], "B")
        self.assertTrue(b.is_live(4, 3, "B"))

    def test_larger_region_captures_several_dots(self):
        b = KropkiBoard(7)
        ring = [(2, 1), (3, 1), (4, 1), (5, 2), (5, 3), (4, 4), (3, 4), (2, 4), (1, 3), (1, 2)]
        inside = [(2, 2), (3, 2), (4, 2), (3, 3), (2, 3), (4, 3)]
        moves = []
        for i, a in enumerate(ring):
            moves.append(a)
            moves.append(inside[i] if i < len(inside) else (6, i - len(inside)))
        play_all(b, moves[:-2])          # everything but the closing A move
        self.assertEqual(b.scores["A"], 0)
        mine, _ = b.play(*ring[-1])
        self.assertEqual(mine, len(inside))
        self.assertEqual(b.scores["A"], len(inside))
        for x, y in inside:
            self.assertEqual(b.territory[y][x], "A")
        self.assertFalse(b.is_playable(3, 2))

    def test_undo_restores_captures(self):
        b = self.diamond_capture()
        snapshot = ([r[:] for r in b.dot], [r[:] for r in b.territory], dict(b.scores), b.turn)
        b.play(2, 3)
        self.assertTrue(b.undo())
        self.assertEqual(([r[:] for r in b.dot], [r[:] for r in b.territory], b.scores, b.turn),
                         snapshot)
        self.assertEqual(b.bases, [])

    def test_pass_and_game_end(self):
        b = KropkiBoard(4)
        b.pass_turn()
        self.assertEqual(b.turn, "B")
        self.assertFalse(b.is_over())
        b.pass_turn()
        self.assertTrue(b.is_over())
        b.undo()
        self.assertFalse(b.is_over())
        self.assertEqual(b.passes, 1)

    def test_board_full_ends_game(self):
        b = KropkiBoard(3)
        for y in range(3):
            for x in range(3):
                b.play(x, y)
        self.assertTrue(b.is_over())
        self.assertEqual(len(b.legal_moves()), 0)

    def test_illegal_moves(self):
        b = KropkiBoard(5)
        b.play(1, 1)
        with self.assertRaises(ValueError):
            b.play(1, 1)
        with self.assertRaises(ValueError):
            b.play(5, 0)

    def test_render(self):
        b = self.diamond_capture()
        b.play(2, 3)
        text = b.render()
        self.assertIn("b", text.splitlines()[3])   # captured B dot shown lower-case
        self.assertIn("A: 1", text)


if __name__ == "__main__":
    unittest.main()


class KropkiAiTests(unittest.TestCase):
    def capture_position(self):
        b = KropkiBoard(6)
        play_all(b, [(2, 1), (2, 2), (1, 2), (0, 0), (3, 2), (0, 1)])
        return b   # A to move; (2, 3) captures

    def test_heuristic_takes_capture(self):
        from bases.kropki_ai import HeuristicKropki, SearchKropki
        for cls in (HeuristicKropki, SearchKropki):
            self.assertEqual(cls(seed=0).choose_move(self.capture_position()), (2, 3))

    def test_heuristic_blocks_capture(self):
        from bases.kropki_ai import HeuristicKropki
        b = self.capture_position()
        b.pass_turn()                     # B to move: must block at (2, 3)
        self.assertEqual(HeuristicKropki(seed=0).choose_move(b), (2, 3))

    def test_heuristic_avoids_playing_into_house(self):
        from bases.kropki_ai import HeuristicKropki
        b = KropkiBoard(6)
        play_all(b, [(2, 1), (0, 0), (1, 2), (0, 1), (3, 2), (0, 2), (2, 3)])
        self.assertNotEqual(HeuristicKropki(seed=0).choose_move(b), (2, 2))

    def test_ais_finish_a_game(self):
        from bases.kropki_ai import RandomKropki, SearchKropki
        b = KropkiBoard(6)
        players = {"A": SearchKropki(seed=1, breadth=4), "B": RandomKropki(seed=2)}
        while not b.is_over():
            move = players[b.turn].choose_move(b)
            if move is None:
                b.pass_turn()
            else:
                b.play(*move)
        self.assertGreaterEqual(b.scores["A"], b.scores["B"])


class KropkiGroupTests(unittest.TestCase):
    def test_groups_liberties_and_edge(self):
        from bases.kropki_ai import danger, groups4
        b = KropkiBoard(6)
        play_all(b, [(2, 2), (0, 0), (3, 2), (5, 5)])   # A pair in the middle, B on edges
        groups = groups4(b, "A")
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0].dots), [(2, 2), (3, 2)])
        self.assertEqual(groups[0].liberties,
                         {(1, 2), (4, 2), (2, 1), (3, 1), (2, 3), (3, 3)})
        self.assertFalse(groups[0].touches_edge)
        self.assertTrue(all(g.touches_edge for g in groups4(b, "B")))
        self.assertEqual(danger(b, "B"), 0.0)
        self.assertAlmostEqual(danger(b, "A"), 2 / 6)

    def test_heuristic_puts_enemy_in_atari_rather_than_wandering(self):
        from bases.kropki_ai import HeuristicKropki
        b = KropkiBoard(7)
        play_all(b, [(3, 2), (3, 3), (2, 3), (0, 0)])   # B's (3,3) has 2 liberties left
        move = HeuristicKropki(seed=0).choose_move(b)
        self.assertIn(move, [(4, 3), (3, 4)])
