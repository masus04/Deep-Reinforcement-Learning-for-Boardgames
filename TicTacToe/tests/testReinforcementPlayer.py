import unittest

import TicTacToe.config as config
import TicTacToe.players.reinforcePlayer as reinforcePlayer
from TicTacToe.environment.board import TicTacToeBoard
from TicTacToe.players.base_players import RandomPlayer
from TicTacToe.environment.game import TicTacToe


class TestReinforcePlayer(unittest.TestCase):

    def testCreateReinforcementPlayer(self):
        reinforcePlayer.ReinforcePlayer(strategy=reinforcePlayer.PGStrategy, lr=0.001)

    def testDummyForwardPass(self):
        board = TicTacToeBoard()
        value_function = reinforcePlayer.PGStrategy(lr=0.001, batch_size=1)
        value_function.evaluate(board.board)

    def testDummyUpdate(self):
        board = TicTacToeBoard()
        value_function = reinforcePlayer.PGStrategy(lr=0.001, batch_size=1)
        value_function.evaluate(board.board)

        move = RandomPlayer.get_move(board)
        board.apply_move(move, config.BLACK)
        value_function.evaluate(board.board)

        move = RandomPlayer.get_move(board)
        board.apply_move(move, config.WHITE)
        value_function.evaluate(board.board)

    def testDummyTrainReinforcePlayer(self):
        player1 = reinforcePlayer.ReinforcePlayer(strategy=reinforcePlayer.PGStrategy, lr=0.001)
        player2 = RandomPlayer()

        simulation = TicTacToe([player1, player2])
        simulation.run_simulations(10)


if __name__ == '__main__':
    unittest.main()
