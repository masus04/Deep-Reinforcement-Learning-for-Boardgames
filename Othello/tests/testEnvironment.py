import unittest
import numpy as np
import random
import os
from datetime import datetime

import Othello.config as config
# from Othello.environment.game import TicTacToe
from Othello.environment.board import OthelloBoard
# import Othello.players.basePlayers as ttt_players
# from Othello.players.reinforcePlayer import FCReinforcePlayer
# from Othello.environment.evaluation import evaluate_against_base_players
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

    def test_Board_ApplyIllegalMove(self):
        board = OthelloBoard()
        board.apply_move((2,3), config.BLACK)
        self.assertEqual(board.illegal_move, None)

        board.apply_move((1, 1), config.BLACK)
        self.assertEqual(board.illegal_move, config.BLACK)

    def test_Board_GameWon(self):
        board = OthelloBoard()
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
            board = OthelloBoard()
            inverse_board = OthelloBoard()
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
        board = OthelloBoard()
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((1, 1), config.WHITE)
        board.apply_move((2, 2), config.BLACK)

        board.apply_move((1, 2), config.WHITE)
        board.apply_move((1, 0), config.BLACK)

        self.assertEqual((3, 2), board.count_stones())

if __name__ == '__main__':
    unittest.main()
