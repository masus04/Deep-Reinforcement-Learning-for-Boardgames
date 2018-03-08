import unittest
import numpy as np
import random
import os
from datetime import datetime

import TicTacToe.config as config
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.board import TicTacToeBoard
import TicTacToe.players.basePlayers as ttt_players
from TicTacToe.players.reinforcePlayer import ReinforcePlayer, PGStrategy
from TicTacToe.environment.evaluation import evaluate_against_base_players
from plotting import Plotter


class TestEnvironment(unittest.TestCase):

    TEST_EPISODES = 20

    def test_Board_ApplyValidMoves(self):
        board = TicTacToeBoard()
        self.assertEqual(set(board.get_valid_moves(config.BLACK)), set([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]), msg="Valid moves incorrect")
        self.assertEqual(set(board.get_valid_moves(config.WHITE)), set([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]), msg="Valid moves incorrect")
        board.apply_move((1, 1), config.BLACK)
        self.assertEqual(set(board.get_valid_moves(config.BLACK)), set([(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]), msg="Valid moves incorrect")
        self.assertEqual(set(board.get_valid_moves(config.WHITE)), set([(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]), msg="Valid moves incorrect")

    def test_Board_ApplyIllegalMove(self):
        board = TicTacToeBoard()
        board.apply_move((1, 1), config.BLACK)
        self.assertEqual(board.illegal_move, None)

        board.apply_move((1, 1), config.BLACK)
        self.assertEqual(board.illegal_move, config.BLACK)

    def test_Board_GameWon(self):
        board = TicTacToeBoard()
        self.assertFalse(board.game_won(), msg="Empty Board")
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((1, 1), config.WHITE)
        board.apply_move((1, 0), config.BLACK)
        board.apply_move((2, 2), config.WHITE)
        self.assertFalse(board.game_won(), msg="No Winner yet")
        board.apply_move((2, 0), config.BLACK)
        self.assertTrue(board.game_won(), msg="Black Won")

    def test_Board_Representation(self):
        random_player = ttt_players.RandomPlayer()
        boards = []
        inverses = []
        for i in range(100):
            board = TicTacToeBoard()
            inverse_board = TicTacToeBoard()
            for j in range(9):
                move = random_player.get_move(board)
                color = (config.BLACK, config.WHITE)
                color = random.choice(color)

                board.apply_move(move, color)
                boards.append(board.copy())

                inverse_board.apply_move(move, board.other_color(color))
                inverses.append((inverse_board.copy()))

        for i in range(len(boards)):
            rep = boards[i].get_representation(config.WHITE)
            self.assertTrue((rep == inverses[i].board).all(), msg="Inverting board failed")

    def test_Board_CountStones(self):
        board = TicTacToeBoard()
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((1, 1), config.WHITE)
        board.apply_move((2, 2), config.BLACK)

        board.apply_move((1, 2), config.WHITE)
        board.apply_move((1, 0), config.BLACK)

        self.assertEqual((3, 2), board.count_stones())

    def testRandomPlayer(self):
        player1 = ttt_players.RandomPlayer()
        player2 = ttt_players.RandomPlayer()
        simulation = TicTacToe([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        self.assertTrue(len(results) == self.TEST_EPISODES)
        self.assertTrue(None not in results)

        for i in range(4):
            simulation.__run__(player1, player2)
            black_stones, white_stones = simulation.board.count_stones()
            self.assertIn(black_stones, [white_stones-1, white_stones, white_stones+1])
            if not simulation.board.game_won():
                self.assertEqual(black_stones+white_stones, simulation.board.board_size**2)
        print("Average Result Random vs Random: %s" % np.mean(results))

    def test_ExperiencedPlayer(self):
        player1 = ttt_players.ExperiencedPlayer()
        player2 = ttt_players.ExperiencedPlayer()
        simulation = TicTacToe([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        self.assertTrue(len(results) == self.TEST_EPISODES)
        self.assertTrue(None not in results)

        for i in range(4):
            simulation.__run__(player1, player2)
            black_stones, white_stones = simulation.board.count_stones()
            self.assertIn(black_stones, [white_stones-1, white_stones, white_stones+1])
            if not simulation.board.game_won():
                self.assertEqual(black_stones+white_stones, simulation.board.board_size**2)
        print("Average Result Experienced vs Experienced: %s" % np.mean(results))

    def test_ExperiencedVsRandom(self):
        player1 = ttt_players.ExperiencedPlayer()
        player2 = ttt_players.RandomPlayer()
        simulation = TicTacToe([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        print("Average Result Experienced vs Random: %s" % np.mean(results))

    def test_ExperiencedBlockVsRandom(self):
        player1 = ttt_players.ExperiencedPlayer(block_mid=True)
        player2 = ttt_players.RandomPlayer()
        simulation = TicTacToe([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        print("Average Result Experienced(block) vs Random: %s" % np.mean(results))

    def test_BlockMid(self):
        player1 = ttt_players.ExperiencedPlayer(block_mid=True)
        player2 = ttt_players.ExperiencedPlayer(block_mid=False)
        simulation = TicTacToe([player1, player2])
        results, losses = simulation.run_simulations(self.TEST_EPISODES)
        print("Average Result Experienced(block) vs Experienced(): %s" % np.mean(results))

    def test_dynamic_plotting(self):
        plotter = Plotter()
        max = 3000
        for i in range(max):
            plotter.add_values([("loss", (max-i)/max), ("evaluation score", i/max/2), ("second score", 0.3)])

        plotter.plot("DynamicTestPlot").savefig("DynamicTestPlot")
        self.assertTrue(os.path.exists("DynamicTestPlot.png"))

    def test_Evaluation(self):
        start = datetime.now()
        score, results = evaluate_against_base_players(ttt_players.RandomPlayer())
        print("Evaluating RandomPlayer -> score: %s, took: %s" % (score, datetime.now() - start))

        start = datetime.now()
        score, results = evaluate_against_base_players(ttt_players.ExperiencedPlayer())
        print("Evaluating ExpertPlayer -> score: %s, took: %s" % (score, datetime.now() - start))

    def test_Performance(self):
        p1 = ttt_players.RandomPlayer()
        p2 = ttt_players.RandomPlayer()
        simulation = TicTacToe([p1, p2])
        N = 15000

        start = datetime.now()
        simulation.run_simulations(N)
        print("Simulating %s random games took %s" % (N, datetime.now()-start))

    def test_evaluation(self):
        p1 = ttt_players.RandomPlayer()
        evaluate_against_base_players(p1, silent=False)

        p2 = ReinforcePlayer(strategy=PGStrategy(lr=1e-5, batch_size=1))
        evaluate_against_base_players(p2, silent=False)


if __name__ == '__main__':
    unittest.main()
