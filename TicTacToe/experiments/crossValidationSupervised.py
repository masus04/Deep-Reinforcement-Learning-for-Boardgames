import os
from random import uniform

from experiment import Experiment
from TicTacToe.experiments.trainPGStrategySupervised import TrainPGStrategySupervised
from TicTacToe.experiments.trainPGSupervisedContinuous import TrainPGSupervisedContinuous


class SupervisedCrossValidation(Experiment):

    def __init__(self, nested_experiment):
        super(SupervisedCrossValidation, self).__init__(os.path.dirname(os.path.abspath(__file__)))
        self.nested_experiment = nested_experiment

    def reset(self):
        self.nested_experiment.reset()
        return self

    def run(self, iterations, lr_lower_boundary, lr_upper_boundary):

        for i in range(iterations):
            LR = 10 ** uniform(lr_lower_boundary, lr_upper_boundary)

            print("\nIteration %s/% - lr: %s" % (i+1, iterations, LR))

            self.nested_experiment.reset().run(lr=LR)


if __name__ == '__main__':

    AVAILABLE_EXPERIMENTS = TrainPGStrategySupervised, TrainPGSupervisedContinuous

    GAMES = 2
    EPISODES = 40000 // GAMES

    # experiment = SupervisedCrossValidation(TrainPGStrategySupervised(games=GAMES, episodes=EPISODES))
    experiment = SupervisedCrossValidation(TrainPGSupervisedContinuous(games=GAMES * EPISODES))
    experiment.run(5, -4.5, -5)
