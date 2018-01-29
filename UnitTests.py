import unittest

import TicTacToe.config as config
from TicTacToe.environment.board import TicTacToeBoard, BoardException


class TestStringMethods(unittest.TestCase):

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
