import unittest
import numpy as np
import torch

import TicTacToe.config as config
from TicTacToe.environment.board import TicTacToeBoard
from TicTacToe.players.basePlayers import RandomPlayer
from TicTacToe.environment.game import TicTacToe

from TicTacToe.players.searchPlayer import SearchPlayer

CORNERS = [(0, 0), (0, 2), (2, 0), (2, 2)]
SIDES   = [(1, 0), (0, 1), (1, 2), (2, 1)]


class TestSearchPlayer(unittest.TestCase):

    def test_createPlayer(self):
        SearchPlayer()

    def test_neverLose(self):
        GAMES = 1000

        player1 = SearchPlayer()
        player2 = SearchPlayer()
        random_player = RandomPlayer()

        simulation = TicTacToe([player1, player2])
        results, losses = simulation.run_simulations(GAMES)
        self.assertEqual(len(results), results.count(0), "Perfect player mirror match resulted in a result other than draw")
        print("\nFirst 20 results: %s against self" % results[:20])

        simulation = TicTacToe([player1, random_player])
        results, losses = simulation.run_simulations(GAMES)
        self.assertEqual(0, results.count(-1), "Perfect player lost against random")
        print("First 20 results: %s against random player" % results[:20])


if __name__ == '__main__':
    unittest.main()
