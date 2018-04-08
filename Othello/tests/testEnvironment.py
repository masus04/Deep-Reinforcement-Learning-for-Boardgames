import unittest
import numpy as np
import random
import os
from datetime import datetime

import Othello.config as config
from Othello.environment.board import OthelloBoard
from Othello.environment.game import Othello
from Othello.players.basePlayers import RandomPlayer, NovicePlayer, ExperiencedPlayer, ExpertPlayer, SearchPlayer
from Othello.experiments.OthelloBaseExperiment import OthelloBaseExperiment
# from Othello.players.reinforcePlayer import FCReinforcePlayer
from Othello.environment.evaluation import evaluate_against_base_players
from plotting import Plotter


class TestEnvironment(unittest.TestCase):

    TEST_EPISODES = 20

    def test_Board_ApplyValidMoves(self):
        board = OthelloBoard()
        self.assertEqual(board.get_valid_moves(config.BLACK), {(2, 3), (3, 2), (4, 5), (5, 4)}, msg="Valid moves incorrect")
        self.assertEqual(board.get_valid_moves(config.WHITE), {(2, 4), (4, 2), (3, 5), (5, 3)}, msg="Valid moves incorrect")
        board.apply_move((3, 2), config.BLACK)
        self.assertEqual(board.get_valid_moves(config.BLACK), {(4, 5), (5, 4), (5, 5)}, msg="Valid moves incorrect")
        self.assertEqual(board.get_valid_moves(config.WHITE), {(4, 2), (2, 4), (2, 2)}, msg="Valid moves incorrect")

        board.apply_move((2, 2), config.WHITE)
        board.apply_move((1, 2), config.BLACK)
        board.apply_move((1, 1), config.WHITE)
        board.apply_move((1, 0), config.BLACK)
        board.apply_move((0, 0), config.WHITE)
        board.apply_move((4, 5), config.BLACK)
        board.apply_move((5, 5), config.WHITE)
        board.apply_move((6, 5), config.BLACK)
        board.apply_move((6, 6), config.WHITE)
        board.apply_move((6, 7), config.BLACK)
        board.apply_move((7, 7), config.WHITE)

    def test_Board_ApplyIllegalMove(self):
        board = OthelloBoard()
        board.apply_move((2,3), config.BLACK)
        self.assertEqual(board.illegal_move, None)

        board.apply_move((1, 1), config.BLACK)
        self.assertEqual(board.illegal_move, config.BLACK)

    def test_Board_GameWon(self):
        board = OthelloBoard()
        self.assertIsNone(board.game_won(), msg="Empty Board")
        board.apply_move((2, 3), config.BLACK)
        board.apply_move((2, 2), config.WHITE)
        board.apply_move((2, 1), config.BLACK)
        board.apply_move((1, 1), config.WHITE)
        board.apply_move((5, 4), config.BLACK)
        board.apply_move((5, 5), config.WHITE)
        board.apply_move((5, 6), config.BLACK)
        board.apply_move((6, 6), config.WHITE)
        self.assertIsNone(board.game_won(), msg="Empty Board")

        for col in range(len(board.board)):
            for tile in range(len(board.board)):
                if board.board[col, tile] == config.EMPTY:
                    board.board[col, tile] = config.BLACK
        self.assertEqual(board.game_won(), config.BLACK, msg="Black wins by stone count")

        board = OthelloBoard()
        board.apply_move((3, 2), config.BLACK)
        board.apply_move((4, 5), config.BLACK)
        self.assertEqual(board.game_won(), config.BLACK, msg="Black wins by stone count after no players could perform any legal moves")

    def test_Board_Representation(self):
        iterator = OthelloBaseExperiment.AlternatingColorIterator()
        boards = []
        inverses = []

        for i in range(10):
            board = OthelloBoard()
            inverse_board = OthelloBoard()
            inverse_board.board[3, 3] = config.BLACK
            inverse_board.board[3, 4] = config.WHITE
            inverse_board.board[4, 4] = config.BLACK
            inverse_board.board[4, 3] = config.WHITE
            for j in range(30):
                color = iterator.__next__()
                legal_moves = board.get_valid_moves(color)
                if legal_moves:
                    move = random.choice(list(legal_moves))

                    board.apply_move(move, color)
                    boards.append(board.copy())

                    inverse_board.apply_move(move, board.other_color(color))
                    inverses.append((inverse_board.copy()))

        for i in range(len(boards)):
            rep = boards[i].get_representation(config.WHITE)
            self.assertTrue((rep == inverses[i].board).all(), msg="Inverting board failed")

    def test_Board_CountStones(self):
        board = OthelloBoard()
        self.assertEqual((2, 2), board.count_stones())

        board.apply_move((2, 3), config.BLACK)
        board.apply_move((2, 2), config.WHITE)
        board.apply_move((2, 1), config.BLACK)
        board.apply_move((1, 1), config.WHITE)

        self.assertEqual((4, 4), board.count_stones())

    def test_ExperiencedVsRandom(self):
        player1 = ExperiencedPlayer()
        player2 = RandomPlayer()
        simulation = Othello([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        print("Average Result Experienced vs Random: %s" % np.mean(results))

    def testRandomPlayer(self):
        player1 = RandomPlayer()
        player2 = RandomPlayer()
        simulation = Othello([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        self.assertTrue(len(results) == self.TEST_EPISODES)
        self.assertTrue(None not in results)

        print("Average Result Random vs Random: %s" % np.mean(results))

    def test_ExperiencedPlayer(self):
        player1 = ExperiencedPlayer()
        player2 = ExperiencedPlayer()
        simulation = Othello([player1, player2])
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
        simulation = Othello([p1, p2])
        N = 500

        start = datetime.now()
        simulation.run_simulations(N)
        print("Simulating %s random games took %s" % (N, datetime.now()-start))

    # TODO: Rename to execute this test
    def evaluation(self):
        p1 = RandomPlayer()
        evaluate_against_base_players(p1, silent=False)

        p2 = FCReinforcePlayer(lr=1e-5, batch_size=1)
        evaluate_against_base_players(p2, silent=False)


if __name__ == '__main__':
    unittest.main()
