import unittest

from bases.agent import MinimaxAgent, QLearningAgent
from bases.board import Board
from bases.search import negamax, prior_leaf, search_values
from tests.solver import OptimalAgent


class SearchTests(unittest.TestCase):
    def test_full_depth_search_equals_exact_solver(self):
        for extra in (False, True):
            solver = OptimalAgent(2, extra_turn_on_box=extra)
            board = Board(2, extra_turn_on_box=extra)
            for move in [(0, 0, 1, 0), (1, 0, 2, 0), (0, 1, 0, 2)]:
                board.make_move(*move)
            depth = board.geom.num_lines
            self.assertEqual(negamax(board, depth, prior_leaf), solver.value(board.mask))
            self.assertEqual(search_values(board, depth, prior_leaf),
                             solver.action_values(board))
            self.assertEqual(len(board.history), 3, "search must leave the board untouched")

    def test_alpha_beta_matches_plain_negamax(self):
        board = Board(2)
        board.make_move(0, 0, 1, 0)
        board.make_move(0, 0, 0, 1)
        plain = {}
        mover = board.turn
        for a in board.available():
            cap = board.play_index(a)
            child = negamax(board, 4, prior_leaf, -1e9, 1e9)
            plain[a] = cap + (child if board.turn == mover else -child)
            board.undo()
        self.assertEqual(search_values(board, 5, prior_leaf), plain)

    def test_minimax_agent_takes_box(self):
        board = Board(2)
        for m in [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1)]:
            board.make_move(*m)
        self.assertEqual(MinimaxAgent(2, seed=0).choose_action(board), (0, 1, 1, 1))

    def test_q_agent_with_full_search_plays_perfectly(self):
        """From a mid-game position an exhaustive search must reach the exact
        game value against a perfect opponent, whichever side it plays."""
        solver = OptimalAgent(2)
        opening = [(0, 0, 1, 0), (1, 1, 2, 1), (0, 1, 0, 2), (2, 0, 2, 1)]
        for plies in (3, 4):
            board = Board(2)
            for move in opening[:plies]:
                board.make_move(*move)
            side = board.turn
            expected = solver.value(board.mask)
            agent = QLearningAgent(2, training_mode=False, search_depth=12, seed=0)
            players = {side: agent, "A" if side == "B" else "B": solver}
            while not board.is_full():
                board.play_index(players[board.turn].choose_index(board))
            self.assertEqual(board.margin(side), expected)

    def test_noise_makes_random_moves(self):
        board = Board(3)
        for m in [(0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 1, 1)]:
            board.make_move(*m)
        closing = board.geom.index_of((0, 1, 1, 1))
        exact = QLearningAgent(3, training_mode=False, seed=0)
        self.assertTrue(all(exact.choose_index(board) == closing for _ in range(20)))
        noisy = QLearningAgent(3, training_mode=False, noise=1.0, seed=0)
        self.assertLess(sum(noisy.choose_index(board) == closing for _ in range(40)), 40)


if __name__ == "__main__":
    unittest.main()
