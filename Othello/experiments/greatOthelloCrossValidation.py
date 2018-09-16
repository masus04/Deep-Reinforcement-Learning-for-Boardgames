from random import random

from Othello.experiments.othelloBaseExperiment import OthelloBaseExperiment

from Othello.experiments.trainACPlayerVsTraditionalOpponent import TrainACPlayerVsTraditionalOpponent
from Othello.experiments.trainBaselinePlayerVsTraditionalOpponent import TrainBaselinePlayerVsTraditionalOpponent
from Othello.experiments.trainReinforcePlayerVsTraditionalOpponent import TrainReinforcePlayerVsTraditionalOpponent

from Othello.experiments.trainACPlayerVsBest import TrainACPlayerVsBest
from Othello.experiments.trainBaselinePlayerVsBest import TrainBaselinePlayerVsBest
from Othello.experiments.trainReinforcePlayerVsBest import TrainReinforcePlayerVsBest

# from Othello.experiments.trainACPlayerVsSelf import TrainACPlayerVsSelf
from Othello.experiments.trainBaselinePlayerVsSelf import TrainBaselinePlayerVsSelf
# from Othello.experiments.TrainReinforcePlayerVsSelf import TrainReinforcePlayerVsSelf

from Othello.players.reinforcePlayer import FCReinforcePlayer, LargeFCReinforcePlayer, HugeFCReinforcePlayer, ConvReinforcePlayer
from Othello.players.baselinePlayer import FCBaselinePlayer, LargeFCBaselinePlayer, HugeFCBaselinePlayer, ConvBaselinePlayer
from Othello.players.acPlayer import FCACPlayer, LargeFCACPlayer, HugeFCACPlayer, ConvACPlayer


class GreatOthelloCrossValidation(OthelloBaseExperiment):

    def __init__(self, ac, baseline, reinforce, vs_traditional, vs_best, vs_self):
        super(GreatOthelloCrossValidation, self).__init__()
        self.ac = ac
        self.baseline=baseline
        self.reinforce = reinforce
        self.vs_traditional = vs_traditional
        self.vs_best = vs_best
        self.vs_self = vs_self

    def reset(self):
        super().__init__()

    def run(self, lr, games, evaluations):

        if self.vs_traditional:
            PLAYER = None
            OPPONENT = None

            # ACTOR CRITIC
            if self.ac:
                for player in [FCACPlayer(LR), LargeFCACPlayer(LR)]:
                    experiment = TrainACPlayerVsTraditionalOpponent(games=games, evaluations=evaluations, pretrained_player=PLAYER, opponent=OPPONENT)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # BASELINE
            if self.baseline:
                for player in [FCBaselinePlayer(LR), LargeFCBaselinePlayer(LR)]:
                    experiment = TrainBaselinePlayerVsTraditionalOpponent(games=games, evaluations=evaluations, pretrained_player=PLAYER, opponent=OPPONENT)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # REINFORCE
            if self.reinforce:
                for player in [FCReinforcePlayer(LR), LargeFCReinforcePlayer(LR)]:
                    experiment = TrainReinforcePlayerVsTraditionalOpponent(games=games, evaluations=evaluations, pretrained_player=PLAYER, opponent=OPPONENT)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()
            
        if self.vs_best:
            # ACTOR CRITIC
            if self.ac:
                for player in [FCACPlayer(LR), LargeFCACPlayer(LR)]:
                    experiment = TrainACPlayerVsBest(games=games, evaluations=evaluations, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR, milestones=False)
                    experiment.reset()
                    experiment.run(lr=LR, milestones=True)

            # BASELINE
            if self.baseline:
                for player in [FCBaselinePlayer(LR), LargeFCBaselinePlayer(LR)]:
                    experiment = TrainBaselinePlayerVsBest(games=games, evaluations=evaluations, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR, milestones=False)
                    experiment.reset()
                    experiment.run(lr=LR, milestones=True)

            # REINFORCE
            if self.reinforce:
                for player in [FCReinforcePlayer(LR), LargeFCReinforcePlayer(LR)]:
                    experiment = TrainReinforcePlayerVsBest(games=games, evaluations=evaluations, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR, milestones=False)
                    experiment.reset()
                    experiment.run(lr=LR, milestones=True)
        
        if self.vs_self:
            # ACTOR CRITIC
            """ Not implemented 
            if self.ac:
                for player in [FCACPlayer(LR), LargeFCACPlayer(LR)]:
                    experiment = TrainACPlayerVsSelf(games=games, evaluations=evaluations, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()
            """

            # BASELINE
            if self.baseline:
                for player in [FCBaselinePlayer(LR), LargeFCBaselinePlayer(LR)]:
                    experiment = TrainBaselinePlayerVsSelf(games=games, evaluations=evaluations, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()

            # REINFORCE
            """ Not implemented 
            if self.reinforce:
                for player in [FCReinforcePlayer(LR), LargeFCReinforcePlayer(LR)]:
                    experiment = TrainReinforcePlayerVsSelf(games=games, evaluations=evaluations, pretrained_player=player)
                    print("\n|| ----- Running %s with %s ----- ||" % (experiment, player))
                    experiment.run(lr=LR)
                    experiment.reset()
            """


if __name__ == '__main__':

    # Train following players:
    AC = True
    BASELINE = True
    REINFORCE = True

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
        greatCrossVal = GreatOthelloCrossValidation(ac=AC, baseline=BASELINE, reinforce=REINFORCE, vs_traditional=VS_TRADITIONAL, vs_best=VS_BEST, vs_self=VS_SELF)
        greatCrossVal.run(lr=LR, games=GAMES, evaluations=EVALUATIONS)

    print("\n| Great TicTacToe Crossvalidation completed |")
