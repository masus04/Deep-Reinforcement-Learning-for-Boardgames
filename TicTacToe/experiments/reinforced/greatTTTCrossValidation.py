from random import random

from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment

from TicTacToe.experiments.reinforced.trainACPlayerVsTraditionalOpponent import TrainACPlayerVsTraditionalOpponent
from TicTacToe.experiments.reinforced.trainBaselinePlayerVsTraditionalOpponent import TrainBaselinePlayerVsTraditionalOpponent
from TicTacToe.experiments.reinforced.trainReinforcePlayerVsTraditionalOpponent import TrainReinforcePlayerVsTraditionalOpponent

from TicTacToe.experiments.reinforced.trainACPlayerVsBest import TrainACPlayerVsBest
from TicTacToe.experiments.reinforced.trainBaselinePlayerVsBest import TrainBaselinePlayerVsBest
from TicTacToe.experiments.reinforced.trainReinforcePlayerVsBest import TrainReinforcePlayerVsBest

# from TicTacToe.experiments.reinforced.trainACPlayerVsSelf import TrainACPlayerVsSelf
from TicTacToe.experiments.reinforced.trainBaselinePlayerVsSelf import TrainBaselinePlayerVsSelf
# from TicTacToe.experiments.reinforced.trainReinforcePlayerVsSelf import TrainReinforcePlayerVsSelf

from TicTacToe.players.reinforcePlayer import FCReinforcePlayer, LargeFCReinforcePlayer, ConvReinforcePlayer
from TicTacToe.players.baselinePlayer import FCBaseLinePlayer, LargeFCBaseLinePlayer, ConvBaseLinePlayer
from TicTacToe.players.acPlayer import FCACPlayer, LargeFCACPlayer, ConvACPlayer


class GreatTTTCrossValidation(TicTacToeBaseExperiment):

    def __init__(self):
        super(GreatTTTCrossValidation, self).__init__()

    def reset(self):
        super().__init__()

    def run(self):
        if VS_TRADITIONAL:
            # ACTOR CRITIC
            if AC:
                for player in [FCACPlayer(LR)]:
                    experiment = TrainACPlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player, opponent=None)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # BASELINE
            if BASELINE:
                for player in [FCBaseLinePlayer(LR)]:
                    experiment = TrainBaselinePlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player, opponent=None)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # REINFORCE
            if REINFORCE:
                for player in [FCReinforcePlayer(LR)]:
                    experiment = TrainReinforcePlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player, opponent=None)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

        if VS_BEST:
            # ACTOR CRITIC
            if AC:
                for player in [FCACPlayer(LR)]:
                    experiment = TrainACPlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # BASELINE
            if BASELINE:
                for player in [FCBaseLinePlayer(LR)]:
                    experiment = TrainBaselinePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()
                    experiment.run(lr=LR, milestones=True)

            # REINFORCE
            if REINFORCE:
                for player in [FCReinforcePlayer(LR)]:
                    experiment = TrainReinforcePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

        if VS_SELF:
            # ACTOR CRITIC
            """ Not yet implemented
            if AC:
                for player in [FCACPlayer(LR)]:
                    experiment = TrainACPlayerVsSelf(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()
            """

            # BASELINE
            if BASELINE:
                for player in [FCBaseLinePlayer(LR)]:
                    experiment = TrainBaselinePlayerVsSelf(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()
                    experiment.run(lr=LR, milestones=True)

            # REINFORCE
            """ Not yet implemented
            if REINFORCE:
                for player in [FCReinforcePlayer(LR)]:
                    experiment = TrainReinforcePlayerVsSelf(games=GAMES, evaluations=EVALUATIONS, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()
            """


if __name__ == '__main__':

    # Player selection
    AC = True
    BASELINE = True
    REINFORCE = True

    # Mode selection
    VS_TRADITIONAL = True
    VS_BEST = True
    VS_SELF = True

    # Training Parameters    
    LR = 1e-3 + random()*1e-9
    GAMES = 10
    EVALUATIONS = GAMES // 5

    for i in range(1):
        greatCrossVal = GreatTTTCrossValidation()
        greatCrossVal.run()

    print("\n| Great TicTacToe Crossvalidation completed |")
