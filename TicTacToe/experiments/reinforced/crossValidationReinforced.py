from random import uniform
from datetime import datetime

from experiment import Experiment
from TicTacToe.players.base_players import ExperiencedPlayer
from TicTacToe.experiments.reinforced.trainReinforcePlayer import TrainReinforcePlayer
from TicTacToe.experiments.reinforced.trainReinforcePlayerVsTraditionalOpponent import TrainReinforcePlayerVsTraditionalOpponent


class ReinforcedCrossValidation(Experiment):

    def __init__(self, nested_experiment, batch_size):
        super(ReinforcedCrossValidation, self).__init__()
        self.nested_experiment = nested_experiment
        self.batch_size = batch_size

    def reset(self):
        self.nested_experiment.reset()
        return self

    def run(self, iterations, lr_lower_boundary, lr_upper_boundary):

        results = []
        for i in range(iterations):
            LR = 10 ** uniform(lr_lower_boundary, lr_upper_boundary)

            print("\nIteration %s/%s" % (i+1, iterations))
            print("Running CrossValidation for %s with lr: %s" % (self.nested_experiment.__class__.__name__, LR))

            self.nested_experiment.reset().run(lr=LR, batch_size=self.batch_size)
            results.append((self.nested_experiment.final_score, LR))

        return sorted(results, reverse=True)


if __name__ == '__main__':

    start = datetime.now()

    GAMES = 100
    EVALUATIONS = 1
    BATCH_SIZE = 32

    PLAYER = None  # PLAYER = Experiment.load_player("ReinforcePlayer using 3 layers pretrained on legal moves for 1000000 games.pth")

    experiment = ReinforcedCrossValidation(TrainReinforcePlayer(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER), BATCH_SIZE)
    #experiment = ReinforcedCrossValidation(TrainReinforcePlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=ExperiencedPlayer), BATCH_SIZE)

    results = experiment.run(5, -4, -5)

    print("\nFinal Reward - LR:")
    for res in results:
        print("%s - %s" % (res[0], res[1]))

    print("\nCrossvalidation complete, took: %s" % (datetime.now() - start))
