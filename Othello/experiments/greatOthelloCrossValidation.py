from random import random

from Othello.experiments.OthelloBaseExperiment import OthelloBaseExperiment
from Othello.experiments.trainACPlayerVsBest import TrainACPlayerVsBest
from Othello.experiments.trainBaselinePlayerVsBest import TrainBaselinePlayerVsBest
from Othello.experiments.trainReinforcePlayerVsBest import TrainReinforcePlayerVsBest
from Othello.players.reinforcePlayer import FCReinforcePlayer, LargeFCReinforcePlayer, ConvReinforcePlayer
from Othello.players.baselinePlayer import FCBaselinePlayer, LargeFCBaselinePlayer, ConvBaselinePlayer
from Othello.players.acPlayer import FCACPlayer, LargeFCACPlayer, ConvACPlayer


class GreatTTTCrossValidation(OthelloBaseExperiment):

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
            experiment.reset()

        # BASELINE
        for player in [FCBaselinePlayer(LR), LargeFCBaselinePlayer(LR)]:
            experiment = TrainBaselinePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
            print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
            experiment.run(lr=LR, batch_size=1)
            experiment.reset()

        # REINFORCE
        for player in [FCReinforcePlayer(LR), LargeFCReinforcePlayer(LR)]:
            experiment = TrainReinforcePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
            print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
            experiment.run(lr=LR, batch_size=1)
            experiment.reset()


if __name__ == '__main__':

    for i in range(1):
        greatCrossVal = GreatTTTCrossValidation()
        greatCrossVal.run()

    print("\n| Great TicTacToe Crossvalidation completed |")
