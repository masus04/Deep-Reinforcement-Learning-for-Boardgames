from abc import ABC
from datetime import datetime

import Othello.config as conf
from Othello.players.basePlayers import RandomPlayer
from Othello.environment.board import OthelloBoard
from experiment import Experiment


class OthelloBaseExperiment(Experiment):
    config = conf

    def __init__(self):
        super(OthelloBaseExperiment, self).__init__()

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
        color_iterator = OthelloBaseExperiment.AlternatingColorIterator()

        start = datetime.now()
        training_set = []
        for game in range(games):
            board = OthelloBoard()
            while board.game_won() is None:
                # generate training pair
                expert_move = labeling_strategy.get_move(board)
                training_set.append((board.copy(), expert_move))

                # prepare for next sample
                color = color_iterator.__next__()
                generator.color = color
                move = generator.get_move(board)
                if move is not None:
                    board.apply_move(move, color)

        print("Generated %s training pairs form %s games in %s" % (len(training_set), games, datetime.now() - start))
        return training_set
