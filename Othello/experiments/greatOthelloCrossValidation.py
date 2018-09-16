from random import random

from Othello.experiments.othelloBaseExperiment import OthelloBaseExperiment

from Othello.experiments.trainACPlayerVsTraditionalOpponent import TrainACPlayerVsTraditionalOpponent
from Othello.experiments.trainBaselinePlayerVsTraditionalOpponent import TrainBaselinePlayerVsTraditionalOpponent
from Othello.experiments.trainReinforcePlayerVsTraditionalOpponent import TrainReinforcePlayerVsTraditionalOpponent

from Othello.experiments.trainACPlayerVsBest import TrainACPlayerVsBest
from Othello.experiments.trainBaselinePlayerVsBest import TrainBaselinePlayerVsBest
from Othello.experiments.trainReinforcePlayerVsBest import TrainReinforcePlayerVsBest

from Othello.experiments.trainACPlayerVsSelf import TrainACPlayerVsSelf
from Othello.experiments.trainBaselinePlayerVsSelf import TrainBaselinePlayerVsSelf
from Othello.experiments.TrainReinforcePlayerVsSelf import TrainReinforcePlayerVsSelf

from Othello.players.reinforcePlayer import FCReinforcePlayer, LargeFCReinforcePlayer, HugeFCReinforcePlayer, ConvReinforcePlayer
from Othello.players.baselinePlayer import FCBaselinePlayer, LargeFCBaselinePlayer, HugeFCBaselinePlayer, ConvBaselinePlayer
from Othello.players.acPlayer import FCACPlayer, LargeFCACPlayer, HugeFCACPlayer, ConvACPlayer

class GreatOthelloCrossValidation(OthelloBaseExperiment):

    def __init__(self):
        super(GreatOthelloCrossValidation, self).__init__()

    def reset(self):
        super().__init__()

    def run(self):

        if VS_TRADITIONAL:
            PLAYER = None
            OPPONENT = None

            # ACTOR CRITIC
            if AC:
                for player in [LargeFCACPlayer(LR), HugeFCACPlayer(LR)]:
                    experiment = TrainACPlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=OPPONENT)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # BASELINE
            if BASELINE:
                for player in [LargeFCBaselinePlayer(LR), HugeFCBaselinePlayer(LR)]:
                    experiment = TrainBaselinePlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=OPPONENT)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # REINFORCE
            if REINFORCE:
                for player in [LargeFCReinforcePlayer(LR), HugeFCReinforcePlayer(LR)]:
                    experiment = TrainReinforcePlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=OPPONENT)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()
            
        if VS_BEST:
            # ACTOR CRITIC
            if AC:
                for player in [LargeFCACPlayer(LR), HugeFCACPlayer(LR)]:
                    experiment = TrainACPlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # BASELINE
            if BASELINE:
                for player in [LargeFCBaselinePlayer(LR), HugeFCBaselinePlayer(LR)]:
                    experiment = TrainBaselinePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # REINFORCE
            if REINFORCE:
                for player in [LargeFCReinforcePlayer(LR), HugeFCReinforcePlayer(LR)]:
                    experiment = TrainReinforcePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()
        
        if VS_SELF:
            # ACTOR CRITIC
            if AC:
                for player in [LargeFCACPlayer(LR), HugeFCACPlayer(LR)]:
                    experiment = TrainACPlayerVsSelf(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # BASELINE
            if BASELINE:
                for player in [LargeFCBaselinePlayer(LR), HugeFCBaselinePlayer(LR)]:
                    experiment = TrainBaselinePlayerVsSelf(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # REINFORCE
            if REINFORCE:
                for player in [LargeFCReinforcePlayer(LR), HugeFCReinforcePlayer(LR)]:
                    experiment = TrainReinforcePlayerVsSelf(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()


if __name__ == '__main__':

    # Train following players:
    AC = False
    BASELINE = True
    REINFORCE = False

    # Train following modes
    VS_TRADITIONAL = True
    VS_BEST = True
    VS_SELF = True

    # Training Parameters
    LR = 1e-3 + random()*1e-9
    GAMES = 1000000
    EVALUATION_PERIOD = 100
    EVALUATIONS = GAMES // EVALUATION_PERIOD

    for i in range(1):
        greatCrossVal = GreatOthelloCrossValidation()
        greatCrossVal.run()

    print("\n| Great TicTacToe Crossvalidation completed |")
