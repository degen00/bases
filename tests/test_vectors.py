"""The committed conformance vectors must replay identically on this engine."""
import json
import unittest
from pathlib import Path

from tools.export_vectors import DEFAULT_OUTPUT, FORMAT, export, verify

VECTORS = Path(DEFAULT_OUTPUT)


class VectorTests(unittest.TestCase):
    def test_export_is_deterministic_and_self_consistent(self):
        a = export(games=3, seed=7)
        b = export(games=3, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a["format"], FORMAT)
        for record in a["games"]:
            self.assertEqual(verify(record), [])

    def test_vectors_contain_captures_and_passes(self):
        data = export(games=8, seed=1)
        moves = [m for g in data["games"] for m in g["moves"]]
        self.assertTrue(any(m["captured"] for m in moves))
        self.assertTrue(any(m.get("pass") for m in moves))

    def test_verify_detects_tampering(self):
        record = export(games=1, seed=5)["games"][0]
        record["final"]["scores"][0] += 1
        self.assertTrue(any("final scores" in p for p in verify(record)))

    @unittest.skipUnless(VECTORS.is_file(), "no committed vector file")
    def test_committed_vectors_replay(self):
        with open(VECTORS, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["format"], FORMAT)
        self.assertGreaterEqual(len(data["games"]), 20)
        for record in data["games"]:
            self.assertEqual(verify(record), [], f"game {record['id']}")


if __name__ == "__main__":
    unittest.main()
