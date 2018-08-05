from datetime import datetime
from abc import ABC

import TicTacToe.config as conf
from TicTacToe.environment.board import TicTacToeBoard
from TicTacToe.players.basePlayers import ExperiencedPlayer, RandomPlayer
from experiment import Experiment


class TicTacToeBaseExperiment(Experiment):
    config = conf

    def __init__(self):
        super(TicTacToeBaseExperiment, self).__init__()

    @classmethod
    def generate_supervised_training_data(cls, games, labeling_strategy):
        """
        Generates training data by applying random moves to a board and labeling each sample with the move that :param labeling_strategy would have taken given the board.

        :param games: The number of games to be simulated
        :param labeling_strategy: The strategy used to label each sample. The label equals labeling_strategy.get_move(board)
        :return: a list of tuples(board_sample, move_label)
        """

        labeling_strategy.color = cls.config.BLACK

        generator = RandomPlayer()
        color_iterator = TicTacToeBaseExperiment.AlternatingColorIterator()

        start = datetime.now()
        training_set = []
        for game in range(games):
            board = TicTacToeBoard()
            for i in range(9):
                # generate training pair
                expert_move = labeling_strategy.get_move(board)
                training_set.append((board.copy(), expert_move))

                # prepare for next sample
                move = generator.get_move(board)
                board.apply_move(move, color_iterator.__next__())

        print("Generated %s training pairs form %s games in %s" % (len(training_set), games, datetime.now() - start))
        return training_set
