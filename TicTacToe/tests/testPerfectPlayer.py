import unittest
import numpy as np
import torch

import TicTacToe.config as config
from TicTacToe.players.perfectPlayer import PerfectPlayer
from TicTacToe.environment.board import TicTacToeBoard
from TicTacToe.players.basePlayers import RandomPlayer
from TicTacToe.environment.game import TicTacToe


class TestReinforcePlayer(unittest.TestCase):

    def test_createReinforcementPlayer(self):
        PerfectPlayer()

    def test_dummyGames(self):
        player1 = PerfectPlayer()
        player1.color = config.BLACK

        player2 = RandomPlayer()
        player1.color = config.WHITE

        simulation = TicTacToe([player1, player2])
        simulation.run_simulations(10)

    def test_win(self):
        perfect_player = PerfectPlayer()
        perfect_player.color = config.BLACK

        board = TicTacToeBoard()
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((0, 1), config.BLACK)
        self.assertEqual(perfect_player.get_move(board), (0, 2), "Did not choose winning move")

        board = TicTacToeBoard()
        board.apply_move((2, 0), config.BLACK)
        board.apply_move((2, 1), config.BLACK)
        self.assertEqual(perfect_player.get_move(board), (2, 2), "Did not choose winning move")

    def test_blockWin(self):
        perfect_player = PerfectPlayer()
        perfect_player.color = config.WHITE

        board = TicTacToeBoard()
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((0, 1), config.BLACK)
        self.assertEqual(perfect_player.get_move(board), (0, 2), "Did not choose winning move")

        board = TicTacToeBoard()
        board.apply_move((2, 0), config.BLACK)
        board.apply_move((2, 1), config.BLACK)
        self.assertEqual(perfect_player.get_move(board), (2, 2), "Did not choose winning move")

    def test_fork(self):
        perfect_player = PerfectPlayer()
        perfect_player.color = config.BLACK

        board = TicTacToeBoard()
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((0, 1), config.WHITE)
        board.apply_move((0, 2), config.BLACK)
        self.assertIn(perfect_player.get_move(board), [(2, 0), (0, 2), (1, 1)], "Did not choose fork move")

        board = TicTacToeBoard()
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((1, 1), config.WHITE)
        board.apply_move((2, 2), config.BLACK)
        self.assertIn(perfect_player.get_move(board), [(2, 0), (0, 2)], "Did not choose fork move")

    def test_blockFork(self):
        perfect_player = PerfectPlayer()
        perfect_player.color = config.WHITE

        board = TicTacToeBoard()
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((0, 1), config.WHITE)
        board.apply_move((0, 2), config.BLACK)
        self.assertIn(perfect_player.get_move(board), [(2, 0), (0, 2), (1, 1)], "Did not choose fork move")

        board = TicTacToeBoard()
        board.apply_move((0, 0), config.BLACK)
        board.apply_move((1, 1), config.WHITE)
        board.apply_move((2, 2), config.BLACK)
        self.assertIn(perfect_player.get_move(board), [(2, 0), (0, 2)], "Did not choose fork move")

    def test_neverLose(self):
        GAMES = 100

        player1 = PerfectPlayer()
        player1.color = config.BLACK

        player2 = RandomPlayer()
        player1.color = config.WHITE

        simulation = TicTacToe([player1, player2])
        results, losses = simulation.run_simulations(GAMES)

        results


if __name__ == '__main__':
    unittest.main()
