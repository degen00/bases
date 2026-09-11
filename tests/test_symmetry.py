import random
import unittest

from bases.board import Board, Geometry


class SymmetryTests(unittest.TestCase):
    def test_permutations_are_bijections_and_identity_first(self):
        for n in (1, 2, 3, 4):
            g = Geometry.get(n)
            self.assertEqual(g.perms[0], list(range(g.num_lines)))
            for perm, inv in zip(g.perms, g.inverse):
                self.assertEqual(sorted(perm), list(range(g.num_lines)))
                self.assertEqual([inv[p] for p in perm], list(range(g.num_lines)))

    def test_eight_distinct_symmetries(self):
        g = Geometry.get(3)
        self.assertEqual(len({tuple(p) for p in g.perms}), 8)

    def test_transform_matches_slow_permute(self):
        rng = random.Random(0)
        for n in (2, 3, 4):
            g = Geometry.get(n)
            for _ in range(100):
                m = rng.getrandbits(g.num_lines)
                for k in range(8):
                    self.assertEqual(g.transform(m, k), g.permute(m, g.perms[k]))

    def test_symmetry_preserves_box_structure(self):
        """A completed box stays a completed box under every symmetry."""
        g = Geometry.get(3)
        for b, bm in enumerate(g.box_masks):
            for k in range(8):
                self.assertIn(g.transform(bm, k), g.box_masks)

    def test_canonical_is_invariant(self):
        rng = random.Random(1)
        for n in (2, 3):
            g = Geometry.get(n)
            for _ in range(200):
                m = rng.getrandbits(g.num_lines)
                cm, ks = g.canonical(m)
                for k in ks:
                    self.assertEqual(g.transform(m, k), cm)
                self.assertEqual(len(ks), sum(g.transform(m, k) == cm for k in range(8)))
                for j in range(8):
                    self.assertEqual(g.canonical(g.transform(m, j))[0], cm)

    def test_canonical_action_mapping_is_consistent(self):
        """(state, action) and its rotated copy must map to the same key."""
        rng = random.Random(2)
        g = Geometry.get(3)
        for _ in range(200):
            m = rng.getrandbits(g.num_lines)
            free = [i for i in range(g.num_lines) if not m >> i & 1]
            if not free:
                continue
            a = rng.choice(free)
            cm, ks = g.canonical(m)
            key = (cm, g.canonical_action(ks, a))
            for j in range(8):
                m2, a2 = g.transform(m, j), g.perms[j][a]
                cm2, ks2 = g.canonical(m2)
                self.assertEqual((cm2, g.canonical_action(ks2, a2)), key)

    def test_symmetric_position_merges_symmetric_actions(self):
        g = Geometry.get(2)
        cm, ks = g.canonical(0)          # the empty board has all 8 symmetries
        self.assertEqual(len(ks), 8)
        keys = {g.canonical_action(ks, a) for a in range(g.num_lines)}
        self.assertEqual(len(keys), 2, "2x2: only edge vs. inner first moves differ")

    def test_rotated_game_gives_rotated_captures(self):
        """Playing a rotated move sequence yields the rotated final position."""
        rng = random.Random(3)
        g = Geometry.get(3)
        for k in range(8):
            a, b = Board(3), Board(3)
            order = list(range(g.num_lines))
            rng.shuffle(order)
            for idx in order:
                a.play_index(idx)
                b.play_index(g.perms[k][idx])
            self.assertEqual(b.mask, g.transform(a.mask, k))
            self.assertEqual(a.scores, b.scores)


if __name__ == "__main__":
    unittest.main()
