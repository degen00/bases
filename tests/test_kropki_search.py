import unittest

from bases.kropki import KropkiBoard
from bases.kropki_ai import (HeuristicKropki, QuickContext, SearchKropki, board_key,
                             capture_map, make_kropki_ai)
from bases.kropki_tournament import play_match
from tests.test_kropki import play_all


class CaptureMapTests(unittest.TestCase):
    def brute_force(self, board, player):
        """Reference: play every point and record real captures."""
        out = {}
        for move in board.legal_moves():
            trial = board.copy()
            trial.turn = player
            mine, _ = trial.play(*move)
            if mine:
                out[move] = mine
        return out

    def test_matches_brute_force_on_random_positions(self):
        import random
        rng = random.Random(11)
        for size in (5, 6, 8):
            for _ in range(25):
                b = KropkiBoard(size)
                for _ in range(rng.randrange(6, size * size - 4)):
                    moves = b.legal_moves()
                    if not moves:
                        break
                    b.play(*rng.choice(moves))
                for player in "AB":
                    caps, reach = capture_map(b, player)
                    self.assertEqual(caps, self.brute_force(b, player),
                                     f"\n{b.render()}\nplayer {player}")

    def test_houses_are_unreachable(self):
        house = KropkiBoard(6)
        play_all(house, [(2, 1), (0, 0), (1, 2), (0, 1), (3, 2), (0, 2), (2, 3)])
        caps, reach = capture_map(house, "A")
        self.assertNotIn((2, 2), reach)
        self.assertEqual(caps, {})
        ctx = QuickContext(house)                     # B to move
        self.assertTrue(ctx.in_enemy_house((2, 2)))
        self.assertFalse(ctx.in_enemy_house((4, 4)))


class QuickScoreTests(unittest.TestCase):
    def test_quick_score_sees_capture_block_and_self_capture(self):
        b = KropkiBoard(6)
        play_all(b, [(2, 1), (2, 2), (1, 2), (0, 0), (3, 2), (0, 1)])   # A to move
        ai = SearchKropki(seed=0)
        ctx = QuickContext(b)
        scores = {m: ai.quick_score(b, m, ctx) for m in b.legal_moves()}
        self.assertEqual(max(scores, key=scores.get), (2, 3), "closing capture ranks first")
        b.pass_turn()                                                    # B to move
        ctx = QuickContext(b)
        scores = {m: ai.quick_score(b, m, ctx) for m in b.legal_moves()}
        self.assertEqual(max(scores, key=scores.get), (2, 3), "block ranks first")
        house = KropkiBoard(6)
        play_all(house, [(2, 1), (0, 0), (1, 2), (0, 1), (3, 2), (0, 2), (2, 3)])  # B to move
        ctx = QuickContext(house)
        self.assertLess(ai.quick_score(house, (2, 2), ctx), ai.quick_score(house, (4, 4), ctx))

    def test_exact_candidates_verify_region_captures(self):
        """A ring of eight A dots around a 2x2 empty house.  A B dot placed
        inside still has free neighbours, so the liberty estimate thinks it is
        safe, but the engine captures it at once; the verified candidates
        must report that loss and rank the move below a safe one."""
        house = KropkiBoard(7)
        ring = [(2, 1), (3, 1), (4, 2), (4, 3), (3, 4), (2, 4), (1, 3), (1, 2)]
        fillers = [(6, y) for y in range(7)]
        moves = []
        for i, a in enumerate(ring):
            moves.append(a)
            if i < len(ring) - 1:
                moves.append(fillers[i])
        play_all(house, moves)                                   # B to move
        self.assertEqual(house.turn, "B")
        self.assertTrue(house.is_playable(2, 2))
        ai = SearchKropki(seed=0)
        ctx = QuickContext(house)
        base, est = ai.quick_parts(house, (2, 2), ctx)
        self.assertGreater(base, -ai.W_LOSS, "liberty estimate sees no danger")
        cands = ai.exact_candidates(house, 40)
        inside = next(c for c in cands if c[1] == (2, 2))
        self.assertEqual(inside[4], 1, "playing inside the house loses the dot")
        self.assertNotEqual(cands[0][1], (2, 2))

    def test_board_key_distinguishes_turn_and_territory(self):
        a, b = KropkiBoard(4), KropkiBoard(4)
        self.assertEqual(board_key(a), board_key(b))
        a.pass_turn()
        self.assertNotEqual(board_key(a), board_key(b))


class SearchTests(unittest.TestCase):
    def test_fixed_depth_is_deterministic_and_legal(self):
        b = KropkiBoard(7)
        play_all(b, [(3, 3), (3, 4), (2, 4), (4, 3)])
        m1 = SearchKropki(seed=1, max_depth=3, time_budget=None).choose_move(b)
        m2 = SearchKropki(seed=1, max_depth=3, time_budget=None).choose_move(b)
        self.assertEqual(m1, m2)
        self.assertIn(m1, b.legal_moves())

    def test_time_budget_is_respected_and_depth_reported(self):
        import time
        b = KropkiBoard(12)
        play_all(b, [(5, 5), (6, 6), (5, 6), (6, 5)])
        ai = SearchKropki(seed=0, max_depth=20, time_budget=0.3)
        t0 = time.perf_counter()
        move = ai.choose_move(b)
        elapsed = time.perf_counter() - t0
        self.assertIn(move, b.legal_moves())
        self.assertLess(elapsed, 1.0)
        self.assertGreaterEqual(ai.depth_reached, 1)

    def test_search_finds_forced_capture(self):
        """B's dot (3,3) has liberties (4,3) and (3,4).  A's dots at (2,4) and
        (4,4) mean that after A plays (4,3), B's only escape (3,4) leaves the
        pair with a single liberty (3,5), so A captures next move.  Playing
        (3,4) first instead lets B run away via (4,3).  Depth 3 sees this."""
        b = KropkiBoard(7)
        play_all(b, [(3, 2), (3, 3), (2, 3), (0, 0), (2, 4), (0, 1), (4, 4), (0, 2)])
        self.assertEqual(b.turn, "A")
        ai = SearchKropki(seed=0, max_depth=3, time_budget=None)
        self.assertEqual(ai.choose_move(b), (4, 3))
        # and the line really is forced: B extends, A captures both dots
        b.play(4, 3)
        b.play(3, 4)
        self.assertEqual(b.play(3, 5), (2, 0))

    def test_levels_factory(self):
        self.assertIsInstance(make_kropki_ai("expert"), SearchKropki)
        self.assertEqual(make_kropki_ai("expert").time_budget, 2.0)
        self.assertIsInstance(make_kropki_ai("normal"), HeuristicKropki)
        self.assertGreater(make_kropki_ai("easy").noise, 0)

    def test_tournament_harness(self):
        result = play_match("normal", "random", 2, 6, 6, seed=3)
        self.assertEqual(result.games, 2)
        self.assertEqual(result.wins + result.losses + result.ties, 2)
        self.assertGreater(result.ms_left, 0)
        self.assertGreaterEqual(result.score, 0.5)


if __name__ == "__main__":
    unittest.main()
