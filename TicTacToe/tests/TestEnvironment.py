import unittest
import numpy as np

import TicTacToe.config as config
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.board import TicTacToeBoard, BoardException
import TicTacToe.players.base_players as ttt_players


class TestEnvironment(unittest.TestCase):

    TEST_EPISODES = 20

    def testBoardValidMoves(self):
        board = TicTacToeBoard()
        self.assertEqual(set(board.get_valid_moves(config.BLACK)), set([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]), msg="Valid moves incorrect")
        self.assertEqual(set(board.get_valid_moves(config.WHITE)), set([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]), msg="Valid moves incorrect")
        board.apply_move((1, 1), config.BLACK)
        self.assertEqual(set(board.get_valid_moves(config.BLACK)), set([(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]), msg="Valid moves incorrect")
        self.assertEqual(set(board.get_valid_moves(config.WHITE)), set([(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]), msg="Valid moves incorrect")

    def testBoardApplyIllegalMove(self):
        board = TicTacToeBoard()
        board.apply_move((1, 1), config.BLACK)
        self.assertRaises(BoardException, board.apply_move, (1, 1), config.BLACK)

    def testBoardGameWon(self):
        board = TicTacToeBoard()
        self.assertFalse(board.game_won(), msg="Empty Board")
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((1, 1), config.WHITE)
        board.apply_move((1, 0), config.BLACK)
        board.apply_move((2, 2), config.WHITE)
        self.assertFalse(board.game_won(), msg="No Winner yet")
        board.apply_move((2, 0), config.BLACK)
        self.assertTrue(board.game_won(), msg="Black Won")

    def testRandomPlayer(self):
        player1 = ttt_players.RandomPlayer()
        player2 = ttt_players.RandomPlayer()
        simulation = TicTacToe(player1, player2)
        results = simulation.run_simulations(self.TEST_EPISODES)
        self.assertTrue(len(results) == self.TEST_EPISODES)
        self.assertTrue(None not in results)

        for i in range(self.TEST_EPISODES):
            simulation.__run__(player1, player2)
            black_stones, white_stones = simulation.board.count_stones()
            self.assertIn(black_stones, [white_stones-1, white_stones, white_stones+1])
            if not simulation.board.game_won():
                self.assertEqual(black_stones+white_stones, simulation.board.board_size**2)
        print("Average Result Random vs Random: %s" % np.mean(results))

    def testExperiencedPlayer(self):
        player1 = ttt_players.ExperiencedPlayer()
        player2 = ttt_players.ExperiencedPlayer()
        simulation = TicTacToe(player1, player2)
        results = simulation.run_simulations(self.TEST_EPISODES)
        self.assertTrue(len(results) == self.TEST_EPISODES)
        self.assertTrue(None not in results)

        for i in range(self.TEST_EPISODES):
            simulation.__run__(player1, player2)
            black_stones, white_stones = simulation.board.count_stones()
            self.assertIn(black_stones, [white_stones-1, white_stones, white_stones+1])
            if not simulation.board.game_won():
                self.assertEqual(black_stones+white_stones, simulation.board.board_size**2)
        print("Average Result Experienced vs Experienced: %s" % np.mean(results))

    def testExperiencedVSRandom(self):
        player1 = ttt_players.ExperiencedPlayer()
        player2 = ttt_players.RandomPlayer()
        simulation = TicTacToe(player1, player2)
        results = simulation.run_simulations(self.TEST_EPISODES)
        print("Average Result Experienced vs Random: %s" % np.mean(results))


if __name__ == '__main__':
    unittest.main()
