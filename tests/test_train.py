import csv
import tempfile
import unittest
from pathlib import Path

from bases.agent import QLearningAgent
from bases.config import load_config
from bases.game import Bases
from bases.train import evaluate, exploration_schedule, hyperparameter_tuning, train_agents
from tests.solver import OptimalAgent


class GameLoopTests(unittest.TestCase):
    def test_learner_observes_every_move(self):
        seen = []

        class Spy(QLearningAgent):
            def observe(self, t):
                seen.append(t)
                super().observe(t)

        agent = Spy(2, seed=0)
        game = Bases(2)
        game.play(agent, agent)
        self.assertEqual(len(seen), 12)
        self.assertTrue(seen[-1].terminal)
        self.assertEqual(sum(t.reward for t in seen), 4)
        self.assertTrue(all(not t.same_mover for t in seen[:-1]))

    def test_lines_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lines.csv"
            game = Bases(1, log_moves=True, lines_log_path=path)
            game.play(QLearningAgent(1, seed=0), QLearningAgent(1, seed=1))
            game.play(QLearningAgent(1, seed=0), QLearningAgent(1, seed=1))
            with open(path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            self.assertEqual({r["game_id"] for r in rows}, {str(game.game_id - 1), str(game.game_id)})


class TrainingTests(unittest.TestCase):
    def test_schedule(self):
        self.assertAlmostEqual(exploration_schedule(0, 100, 0.5, 0.01), 0.5)
        self.assertAlmostEqual(exploration_schedule(80, 100, 0.5, 0.01), 0.01)
        self.assertAlmostEqual(exploration_schedule(99, 100, 0.5, 0.01), 0.01)
        self.assertGreater(exploration_schedule(40, 100, 0.5, 0.01), 0.01)

    def test_training_beats_random_and_greedy_on_2x2(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = Path(tmp) / "p.json"
            log = Path(tmp) / "log.csv"
            agent = train_agents(2, 3000, seed=0, eval_episodes=50, save_path=policy,
                                 log_path=log, quiet=True)
            self.assertTrue(policy.is_file() and log.is_file())
            self.assertEqual(agent.episodes_trained, 3000)
            # 2x2 with alternating turns is a forced 3-1 win for the second
            # player; a perfect player scores ~0.85 vs random and ~0.74 vs greedy.
            rnd = evaluate(agent, "random", 200, seed=1)
            grd = evaluate(agent, "greedy", 200, seed=1)
            self.assertGreater(rnd.score, 0.78, rnd)
            self.assertGreater(grd.score, 0.6, grd)
            # Against the exact solver the agent must win every game it plays
            # as the second player (the theoretically winning side).
            optimal = OptimalAgent(2)
            self.assertEqual(optimal.value(0), -2)
            res = evaluate(agent, optimal, 20, seed=2)
            self.assertEqual((res.wins, res.losses, res.ties), (10, 10, 0), res)
            with open(log) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 10)

    def test_progress_callback_can_stop(self):
        calls = []
        agent = train_agents(2, 1000, seed=0, eval_every=100, eval_episodes=0,
                             progress=lambda s: calls.append(s.episode) or False,
                             save_path=False, log_path=False, quiet=True)
        self.assertEqual(calls, [100])
        self.assertEqual(agent.episodes_trained, 100)

    def test_hyperparameter_tuning_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hp.csv"
            best = hyperparameter_tuning(2, 2, 200, 20, seed=0, results_path=path, quiet=True)
            self.assertIn("learning_rate", best)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)

    def test_config_hyperparameters(self):
        cfg = load_config()
        hp = cfg.hyperparameters(2)
        self.assertEqual(hp["discount_factor"], 1.0)
        self.assertIn("opponent_mix", hp)
        self.assertEqual(cfg.hyperparameters(99)["learning_rate"],
                         cfg.data["training"]["default"]["learning_rate"])


if __name__ == "__main__":
    unittest.main()
