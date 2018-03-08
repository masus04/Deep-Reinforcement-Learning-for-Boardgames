import unittest
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import TicTacToe.config as config
import TicTacToe.players.reinforcePlayer as reinforcePlayer
from TicTacToe.environment.board import TicTacToeBoard
from TicTacToe.players.basePlayers import RandomPlayer
from TicTacToe.environment.game import TicTacToe
from abstractClasses import PlayerException, Model, Strategy


class TestReinforcePlayer(unittest.TestCase):

    def test_CreateReinforcementPlayer(self):
        reinforcePlayer.ReinforcePlayer(strategy=reinforcePlayer.PGStrategy(lr=0.001, batch_size=1))

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
        player1 = reinforcePlayer.ReinforcePlayer(strategy=reinforcePlayer.PGStrategy(lr=0.001, batch_size=1))
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

    def test_discount_rewards(self):
        rewards = [0] * 8 + [1]

        discounted_rewards = Strategy.discount_rewards(rewards, discount_factor=1)
        self.assertEqual(discounted_rewards, [1] * 9)

        discounted_rewards = Strategy.discount_rewards(rewards, discount_factor=0.95)
        self.assertNotEqual(discounted_rewards, rewards)
        self.assertEqual(max(discounted_rewards), discounted_rewards[-1])

    def test_FCReinforcePlayer(self):
        fc_player = reinforcePlayer.FCReinforcePlayer(lr=1e-4)
        random_player = RandomPlayer()

        simulation = TicTacToe([fc_player, random_player])
        simulation.run_simulations(100)

    def test_ConvReinforcePlayer(self):
        fc_player = reinforcePlayer.ConvReinforcePlayer(lr=1e-4)
        random_player = RandomPlayer()

        simulation = TicTacToe([fc_player, random_player])
        simulation.run_simulations(100)


if __name__ == '__main__':
    unittest.main()
