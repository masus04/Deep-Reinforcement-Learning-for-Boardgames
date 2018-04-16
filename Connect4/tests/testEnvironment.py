import unittest
import numpy as np
import random
import os
from datetime import datetime

import Connect4.config as config
from Connect4.environment.board import Connect4Board
from Connect4.environment.game import Connect4
from Connect4.players.basePlayers import RandomPlayer, DeterministicPlayer, NovicePlayer, ExperiencedPlayer, ExpertPlayer
from Connect4.experiments.Connect4BaseExperiment import Connect4BaseExperiment
from Connect4.environment.evaluation import evaluate_against_base_players
from plotting import Plotter


class TestEnvironment(unittest.TestCase):

    TEST_EPISODES = 20

    def test_Board_ApplyValidMoves(self):
        board = Connect4Board()
        self.assertEqual(board.get_valid_moves(config.BLACK), {(5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6)}, msg="Valid moves incorrect")
        self.assertEqual(board.get_valid_moves(config.WHITE), {(5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6)}, msg="Valid moves incorrect")
        board.apply_move((5, 3), config.BLACK)
        self.assertEqual(board.get_valid_moves(config.BLACK), {(5,0), (5,1), (5,2), (4,3), (5,4), (5,5), (5,6)}, msg="Valid moves incorrect")
        self.assertEqual(board.get_valid_moves(config.WHITE), {(5,0), (5,1), (5,2), (4,3), (5,4), (5,5), (5,6)}, msg="Valid moves incorrect")

    def test_Board_ApplyIllegalMove(self):
        board = Connect4Board()
        board.apply_move((5,3), config.BLACK)
        self.assertEqual(board.illegal_move, None)

        board.apply_move((1, 1), config.BLACK)
        print("î Intended illegal move î")
        self.assertEqual(board.illegal_move, config.BLACK)

    def test_Board_GameWon(self):

        # Case 1: Deterministic win
        board = Connect4Board()
        self.assertIsNone(board.game_won(), msg="Empty Board")
        board.apply_move((5, 3), config.BLACK)
        board.apply_move((5, 2), config.WHITE)
        board.apply_move((4, 2), config.BLACK)
        board.apply_move((4, 3), config.WHITE)
        board.apply_move((5, 1), config.BLACK)
        board.apply_move((4, 1), config.WHITE)
        board.apply_move((3, 1), config.BLACK)
        self.assertIsNone(board.game_won(), msg="No Winner yet")

        board.apply_move((5, 0), config.WHITE)
        board.apply_move((4, 0), config.BLACK)
        board.apply_move((3, 0), config.WHITE)
        board.apply_move((2, 0), config.BLACK)
        self.assertEqual(board.game_won(), config.BLACK, msg="Black wins")

    def test_Board_Representation(self):
        board = Connect4Board()
        board.apply_move((5, 3), config.BLACK)
        board.apply_move((5, 2), config.WHITE)
        board.apply_move((4, 2), config.BLACK)
        board.apply_move((4, 3), config.WHITE)
        board.apply_move((5, 1), config.BLACK)
        board.apply_move((4, 1), config.WHITE)
        board.apply_move((3, 1), config.BLACK)
        board.apply_move((5, 0), config.WHITE)
        board.apply_move((4, 0), config.BLACK)
        board.apply_move((3, 0), config.WHITE)
        board.apply_move((2, 0), config.BLACK)

        inv_board = Connect4Board()
        inv_board.apply_move((5, 3), config.WHITE)
        inv_board.apply_move((5, 2), config.BLACK)
        inv_board.apply_move((4, 2), config.WHITE)
        inv_board.apply_move((4, 3), config.BLACK)
        inv_board.apply_move((5, 1), config.WHITE)
        inv_board.apply_move((4, 1), config.BLACK)
        inv_board.apply_move((3, 1), config.WHITE)
        inv_board.apply_move((5, 0), config.BLACK)
        inv_board.apply_move((4, 0), config.WHITE)
        inv_board.apply_move((3, 0), config.BLACK)
        inv_board.apply_move((2, 0), config.WHITE)

        self.assertTrue((board.get_representation(config.WHITE) == inv_board.get_representation(config.BLACK)).all(), "Representationn failed")

    def test_countConnections(self):
        board = Connect4Board()
        board.apply_move((5, 3), config.BLACK)
        board.apply_move((5, 2), config.WHITE)
        board.apply_move((4, 2), config.BLACK)
        board.apply_move((4, 3), config.WHITE)
        board.apply_move((5, 1), config.BLACK)
        board.apply_move((4, 1), config.WHITE)

        self.assertEqual(board.count_connections()[config.BLACK], (2, 2), "Count connections failed")
        self.assertEqual(board.count_connections()[config.WHITE], (2, 2), "Count connections failed")

        board.apply_move((3, 1), config.BLACK)
        board.apply_move((5, 0), config.WHITE)
        board.apply_move((4, 0), config.BLACK)
        board.apply_move((3, 0), config.WHITE)
        board.apply_move((2, 0), config.BLACK)

        self.assertEqual(board.count_connections()[config.BLACK], (4, 1), "Count connections failed")
        self.assertEqual(board.count_connections()[config.WHITE], (3, 1), "Count connections failed")


    def test_ExperiencedVsRandom(self):
        player1 = ExperiencedPlayer()
        player2 = RandomPlayer()
        simulation = Connect4([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        print("Average Result Experienced vs Random: %s" % np.mean(results))

    def testRandomPlayer(self):
        player1 = RandomPlayer()
        player2 = RandomPlayer()
        simulation = Connect4([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        self.assertTrue(len(results) == self.TEST_EPISODES)
        self.assertTrue(None not in results)

        print("Average Result Random vs Random: %s" % np.mean(results))

    def test_ExperiencedPlayer(self):
        player1 = ExperiencedPlayer()
        player2 = ExperiencedPlayer()
        simulation = Connect4([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        self.assertTrue(len(results) == self.TEST_EPISODES)
        self.assertTrue(None not in results)

        print("Average Result Experienced vs Experienced: %s" % np.mean(results))

    def test_dynamic_plotting(self):
        plotter = Plotter()
        max = 3000
        for i in range(max):
            plotter.add_values([("loss", (max-i)/max), ("evaluation score", i/max/2), ("second score", 0.3)])

        plotter.plot("DynamicTestPlot").savefig("DynamicTestPlot")
        self.assertTrue(os.path.exists("DynamicTestPlot.png"))

    def test_Evaluation(self):
        start = datetime.now()
        score, results, overview = evaluate_against_base_players(RandomPlayer())
        print("Evaluating RandomPlayer -> score: %s, took: %s" % (score, datetime.now() - start))

        start = datetime.now()
        score, results, overview = evaluate_against_base_players(ExperiencedPlayer())
        print("Evaluating ExpertPlayer -> score: %s, took: %s" % (score, datetime.now() - start))

    def test_Performance(self):
        p1 = RandomPlayer()
        p2 = RandomPlayer()
        simulation = Connect4([p1, p2])
        N = 500

        start = datetime.now()
        simulation.run_simulations(N)
        print("Simulating %s random games took %s" % (N, datetime.now()-start))


if __name__ == '__main__':
    unittest.main()
