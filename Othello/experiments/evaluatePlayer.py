from random import random
from datetime import datetime

import Othello.config as conf
from Othello.experiments.othelloBaseExperiment import OthelloBaseExperiment
from Othello.environment.evaluation import evaluate_against_each_other
from Othello.players.basePlayers import RandomPlayer, HumanPlayer, SearchPlayer


class EvaluatePlayer(OthelloBaseExperiment):
    config = conf

    def __init__(self):
        super(EvaluatePlayer, self).__init__()

    def reset(self):
        super().__init__()

    def run(self, player1, player2):
        evaluate_against_each_other(player1, player2, games=GAMES, silent=False)


if __name__ == '__main__':

    START_TIME = datetime.now()

    LR = 1e-5 + random() * 1e-9
    PLAYER1 = RandomPlayer()
    PLAYER2 = SearchPlayer(search_depth=5, strategy=0)
    GAMES = 10

    experiment = EvaluatePlayer()
    experiment.run(player1=PLAYER1, player2=PLAYER2)

    print("\n| Evaluation completed, took %s |" % conf.time_diff(START_TIME))
