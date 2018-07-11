from random import random

from Othello.experiments.OthelloBaseExperiment import OthelloBaseExperiment

from Othello.experiments.trainACPlayerVsBest import TrainACPlayerVsBest
from Othello.experiments.trainBaselinePlayerVsBest import TrainBaselinePlayerVsBest
from Othello.experiments.trainReinforcePlayerVsBest import TrainReinforcePlayerVsBest

from Othello.experiments.trainACPlayerVsTraditionalOpponent import TrainACPlayerVsTraditionalOpponent
from Othello.experiments.trainBaselinePlayerVsTraditionalOpponent import TrainBaselinePlayerVsTraditionalOpponent
from Othello.experiments.trainReinforcePlayerVsTraditionalOpponent import TrainReinforcePlayerVsTraditionalOpponent

from Othello.experiments.trainPGSupervisedContinuous import TrainPGSupervisedContinuous

from Othello.players.reinforcePlayer import FCReinforcePlayer, LargeFCReinforcePlayer, HugeFCReinforcePlayer, ConvReinforcePlayer
from Othello.players.baselinePlayer import FCBaselinePlayer, LargeFCBaselinePlayer, HugeFCBaselinePlayer, ConvBaselinePlayer
from Othello.players.acPlayer import FCACPlayer, LargeFCACPlayer, HugeFCACPlayer, ConvACPlayer

BEST = 0
TRADITIONAL = 1
SUPERVISED = 2

class GreatOthelloCrossValidation(OthelloBaseExperiment):

    def __init__(self):
        super(GreatOthelloCrossValidation, self).__init__()

    def reset(self):
        super().__init__()

    def run(self, mode):
        LR = 1e-5 + random()*1e-9
        GAMES = 1000000
        EVALUATION_PERIOD = 10000
        EVALUATIONS = GAMES // EVALUATION_PERIOD

        if mode == BEST:
            # ACTOR CRITIC
            for player in [LargeFCACPlayer(LR), HugeFCACPlayer(LR)]:
                experiment = TrainACPlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                experiment.run(lr=LR)
                experiment.reset()

            # BASELINE
            for player in [LargeFCBaselinePlayer(LR), HugeFCBaselinePlayer(LR)]:
                experiment = TrainBaselinePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                experiment.run(lr=LR)
                experiment.reset()

            # REINFORCE
            for player in [LargeFCReinforcePlayer(LR), HugeFCReinforcePlayer(LR)]:
                experiment = TrainReinforcePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                experiment.run(lr=LR)
                experiment.reset()

        if mode == TRADITIONAL:
            PLAYER = None
            OPPONENT = None

            # ACTOR CRITIC
            for player in [LargeFCACPlayer(LR), HugeFCACPlayer(LR)]:
                experiment = TrainACPlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=OPPONENT)
                print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                experiment.run(lr=LR)
                experiment.reset()

            # BASELINE
            for player in [LargeFCBaselinePlayer(LR), HugeFCBaselinePlayer(LR)]:
                experiment = TrainBaselinePlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=OPPONENT)
                print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                experiment.run(lr=LR)
                experiment.reset()

            # REINFORCE
            for player in [LargeFCReinforcePlayer(LR), HugeFCReinforcePlayer(LR)]:
                experiment = TrainReinforcePlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=OPPONENT)
                print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                experiment.run(lr=LR)
                experiment.reset()

        if mode == SUPERVISED:
            # ACTOR CRITIC
            for player in [LargeFCACPlayer(LR), HugeFCACPlayer(LR)]:
                experiment = TrainPGSupervisedContinuous(games=GAMES, evaluation_period=EVALUATION_PERIOD)
                print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                experiment.run(player=player, lr=LR)
                experiment.reset()

            # BASELINE
            for player in [LargeFCBaselinePlayer(LR), HugeFCBaselinePlayer(LR)]:
                experiment = TrainPGSupervisedContinuous(games=GAMES, evaluation_period=EVALUATION_PERIOD)
                print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                experiment.run(player=player, lr=LR)
                experiment.reset()

            # REINFORCE
            for player in [LargeFCReinforcePlayer(LR), HugeFCReinforcePlayer(LR)]:
                experiment = TrainPGSupervisedContinuous(games=GAMES, evaluation_period=EVALUATION_PERIOD)
                print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                experiment.run(player=player, lr=LR)
                experiment.reset()


if __name__ == '__main__':

    for i in range(1):
        greatCrossVal = GreatOthelloCrossValidation()
        greatCrossVal.run(mode=TRADITIONAL)

    print("\n| Great TicTacToe Crossvalidation completed |")
