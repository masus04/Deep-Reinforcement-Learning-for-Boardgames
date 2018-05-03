from random import random

from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from TicTacToe.experiments.reinforced.trainACPlayerVsBest import TrainACPlayerVsBest
from TicTacToe.experiments.reinforced.trainBaseLinePlayerVsBest import TrainBaselinePlayerVsBest
from TicTacToe.experiments.reinforced.trainReinforcePlayerVsBest import TrainReinforcePlayerVsBest
from TicTacToe.players.reinforcePlayer import FCReinforcePlayer, LargeFCReinforcePlayer, ConvReinforcePlayer
from TicTacToe.players.baselinePlayer import FCBaseLinePlayer, LargeFCBaseLinePlayer, ConvBaseLinePlayer
from TicTacToe.players.acPlayer import FCACPlayer, LargeFCACPlayer, ConvACPlayer


class GreatTTTCrossValidation(TicTacToeBaseExperiment):

    def __init__(self):
        super(GreatTTTCrossValidation, self).__init__()

    def reset(self):
        super().__init__()

    def run(self):
        LR = 1e-5 + random()*1e-9
        GAMES = 200000
        EVALUATIONS = GAMES // 100

        # ACTOR CRITIC
        for player in [FCACPlayer(LR), LargeFCACPlayer(LR)]:
            experiment = TrainACPlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
            print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
            experiment.run(lr=LR, batch_size=1)

        # BASELINE
        for player in [FCBaseLinePlayer(LR), LargeFCBaseLinePlayer(LR)]:
            experiment = TrainBaselinePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
            print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
            experiment.run(lr=LR, batch_size=1)

        # REINFORCE
        for player in [FCReinforcePlayer(LR), LargeFCReinforcePlayer(LR)]:
            experiment = TrainReinforcePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
            print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
            experiment.run(lr=LR, batch_size=1)


if __name__ == '__main__':

    for i in range(1):
        greatCrossVal = GreatTTTCrossValidation()
        greatCrossVal.run()

    print("\n| Great TicTacToe Crossvalidation completed |")
