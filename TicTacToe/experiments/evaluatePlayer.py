from random import random
from datetime import datetime

import TicTacToe.config as conf
from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from TicTacToe.environment.evaluation import evaluate_against_each_other
from TicTacToe.players.basePlayers import RandomPlayer, ExperiencedPlayer, ExpertPlayer


class EvaluatePlayer(TicTacToeBaseExperiment):
    config = conf

    def __init__(self):
        super(EvaluatePlayer, self).__init__()

    def reset(self):
        super().__init__()

    def run(self, player1, player2):
        evaluate_against_each_other(player1, player2, silent=False)


if __name__ == '__main__':

    START_TIME = datetime.now()

    LR = 1e-5 + random() * 1e-9
    PLAYER1 = ExperiencedPlayer()
    PLAYER2 = ExpertPlayer()  # RandomPlayer()

    experiment = EvaluatePlayer()
    experiment.run(player1=PLAYER1, player2=PLAYER2)

    print("\n| Evaluation completed, took %s |" % conf.time_diff(START_TIME))
