import unittest
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import TicTacToe.config as config
import TicTacToe.players.reinforcePlayer as reinforcePlayer
from TicTacToe.environment.board import TicTacToeBoard
from TicTacToe.players.base_players import RandomPlayer
from TicTacToe.environment.game import TicTacToe
from abstractClasses import PlayerException
from abstractClasses import Model


class TestReinforcePlayer(unittest.TestCase):

    def test_CreateReinforcementPlayer(self):
        reinforcePlayer.ReinforcePlayer(strategy=reinforcePlayer.PGStrategy, lr=0.001)

    def test_DummyForwardPass(self):
        board = TicTacToeBoard()
        value_function = reinforcePlayer.PGStrategy(lr=0.001, batch_size=1)
        value_function.evaluate(board.board, board.get_legal_moves_map(config.BLACK))

    def test_DummyUpdate(self):
        board = TicTacToeBoard()
        value_function = reinforcePlayer.PGStrategy(lr=0.001, batch_size=1)
        value_function.evaluate(board.board, board.get_legal_moves_map(config.BLACK))

        move = RandomPlayer.get_move(board)
        board.apply_move(move, config.BLACK)
        value_function.evaluate(board.board, board.get_legal_moves_map(config.BLACK))

        move = RandomPlayer.get_move(board)
        board.apply_move(move, config.WHITE)
        value_function.evaluate(board.board, board.get_legal_moves_map(config.BLACK))

    def test_DummyTrainReinforcePlayer(self):
        player1 = reinforcePlayer.ReinforcePlayer(strategy=reinforcePlayer.PGStrategy, lr=0.001)
        player2 = RandomPlayer()

        simulation = TicTacToe([player1, player2])
        simulation.run_simulations(10)

    def test_LegalSoftMax(self):
        def transform(x):
            return [Variable(torch.FloatTensor((x*3)).view(-1, 9))]

        edge_cases = transform([0.2, 0.3, 0.8])
        edge_cases += transform([-0.2, -0.3, -0.8])
        edge_cases += transform([0.2, -0.3, -0.8])

        edge_cases += transform([20000.3, 30000.3, 80000.3])
        edge_cases += transform([-20000.3, 30000.3, -80000.3])
        edge_cases += transform([20000.3, -30000.3, -80000.3])

        legal_moves = transform([1, 1, 1])
        legal_moves += transform([1, 1, 0])
        legal_moves += transform([1, 0, 0])

        for i, case in enumerate(edge_cases):
            for j, l_moves in enumerate(legal_moves):
                try:
                    x = Model.legal_softmax(case, l_moves)
                except Exception as e:
                    raise PlayerException("LegalSoftMax failed for edge case %s and legal move %s: \n    %s" % (i, j, e))

                self.assertTrue((x == x*l_moves).all(), "LegalSoftMax did not set illegal moves to 0")
                self.assertTrue(x.sum().data[0] > 0, "x.sum <= 0 for edge case %s and legal move %s" % (i, j))
                for elem in x.data.tolist():
                    self.assertNotEqual(elem, np.nan)


if __name__ == '__main__':
    unittest.main()
