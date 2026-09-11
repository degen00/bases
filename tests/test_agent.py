import tempfile
import unittest
from pathlib import Path

from bases.agent import GreedyAgent, QLearningAgent, RandomAgent
from bases.board import Board, Transition


class GreedyAgentTests(unittest.TestCase):
    def test_takes_available_box(self):
        b = Board(2)
        for m in [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1)]:
            b.make_move(*m)
        agent = GreedyAgent(seed=0)
        self.assertEqual(agent.choose_action(b), (0, 1, 1, 1))

    def test_avoids_gift(self):
        b = Board(1)
        b.make_move(0, 0, 1, 0)
        b.make_move(0, 0, 0, 1)
        # Every remaining move gives the box away, so greedy must still move.
        agent = GreedyAgent(seed=0)
        self.assertIn(agent.choose_index(b), b.available())

    def test_random_agent_moves_are_legal(self):
        b = Board(3)
        agent = RandomAgent(seed=0)
        while not b.is_full():
            b.play_index(agent.choose_index(b))
        self.assertTrue(b.is_full())


class QLearningAgentTests(unittest.TestCase):
    def test_update_is_negamax(self):
        agent = QLearningAgent(2, learning_rate=1.0, discount_factor=0.5,
                               use_symmetry=False, use_prior=False, seed=0,
                               backward_replay=False)
        # Opponent-to-move next state with a known best value of 2.
        agent.q[0b11] = {5: 2.0, 6: -1.0}
        agent.observe(Transition(state=0b01, action=1, reward=1.0, next_state=0b11,
                                 next_available=(5, 6), same_mover=False,
                                 terminal=False, player="A"))
        self.assertAlmostEqual(agent.q[0b01][1], 1.0 - 0.5 * 2.0)
        # Same mover next (extra turn): value is added, not subtracted.
        agent.observe(Transition(state=0b01, action=2, reward=0.0, next_state=0b11,
                                 next_available=(5, 6), same_mover=True,
                                 terminal=False, player="A"))
        self.assertAlmostEqual(agent.q[0b01][2], 0.5 * 2.0)
        # Terminal transitions bootstrap nothing.
        agent.observe(Transition(state=0b01, action=3, reward=3.0, next_state=0b11,
                                 next_available=(), same_mover=False,
                                 terminal=True, player="A"))
        self.assertAlmostEqual(agent.q[0b01][3], 3.0)

    def test_backward_replay_propagates_in_one_episode(self):
        agent = QLearningAgent(2, learning_rate=1.0, discount_factor=1.0,
                               use_symmetry=False, use_prior=False, seed=0)
        # s0 -a-> s1 -b-> s2 (terminal, reward 1).  Newest-first replay lets
        # the terminal reward reach Q(s0, a) in the same episode.
        agent.observe(Transition(0b000, 0, 0.0, 0b001, (1,), False, False, "A"))
        self.assertEqual(agent.q, {}, "buffered until the terminal move")
        agent.observe(Transition(0b001, 1, 1.0, 0b011, (), False, True, "B"))
        self.assertAlmostEqual(agent.q[0b001][1], 1.0)
        self.assertAlmostEqual(agent.q[0b000][0], -1.0)

    def test_symmetric_states_share_values(self):
        agent = QLearningAgent(2, learning_rate=1.0, discount_factor=1.0, seed=0)
        b = Board(2)
        b.make_move(0, 0, 1, 0)           # top-left horizontal line
        idx = b.geom.index_of((1, 0, 1, 1))
        agent.observe(Transition(0, b.geom.index_of((0, 0, 1, 0)), 0.0, b.mask,
                                 tuple(b.available()), False, False, "A"))
        agent.observe(Transition(b.mask, idx, 1.0, b.mask | 1 << idx, (), False, True, "B"))
        # The rotated position must give the rotated action the same value.
        b2 = Board(2)
        b2.make_move(0, 0, 0, 1)          # top-left vertical line (rotation of above)
        values = agent.q_values(b2)
        self.assertTrue(agent.knows(b2))
        self.assertEqual(max(values.values()), 1.0)

    def test_choose_action_is_greedy_wrt_q(self):
        agent = QLearningAgent(2, training_mode=False, use_symmetry=False, seed=0)
        b = Board(2)
        agent.q[b.mask] = {3: 0.5, 7: 2.0}
        self.assertEqual(agent.choose_index(b), 7)

    def test_prior_matches_heuristic(self):
        b = Board(2)
        for m in [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1)]:
            b.make_move(*m)
        agent = QLearningAgent(2, seed=0)
        values = agent.q_values(b)
        closing = b.geom.index_of((0, 1, 1, 1))
        self.assertEqual(values[closing], 1.0)
        self.assertEqual(values[b.geom.index_of((1, 0, 2, 0))], 0.0)
        self.assertEqual(values[b.geom.index_of((1, 1, 2, 1))], 0.0)
        self.assertFalse(agent.knows(b))

    def test_unseen_state_falls_back_to_greedy(self):
        agent = QLearningAgent(2, training_mode=False, seed=0)
        b = Board(2)
        for m in [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1)]:
            b.make_move(*m)
        self.assertEqual(agent.choose_action(b), (0, 1, 1, 1))

    def test_save_and_load_roundtrip(self):
        agent = QLearningAgent(2, seed=0)
        agent.q = {5: {1: 0.25, 2: -1.5}, 9: {0: 3.0}}
        agent.episodes_trained = 42
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("policy.json", "policy.json.gz"):
                path = Path(tmp) / name
                agent.save_policy(path)
                loaded = QLearningAgent.load(path)
                self.assertEqual(loaded.q, agent.q)
                self.assertEqual(loaded.episodes_trained, 42)
                self.assertFalse(loaded.training_mode)
                self.assertEqual(QLearningAgent.read_metadata(path)["states"], 2)
                with self.assertRaises(ValueError):
                    QLearningAgent(3).load_policy(path)
        with self.assertRaises(FileNotFoundError):
            QLearningAgent(2).load_policy("/nonexistent/policy.json")


if __name__ == "__main__":
    unittest.main()
