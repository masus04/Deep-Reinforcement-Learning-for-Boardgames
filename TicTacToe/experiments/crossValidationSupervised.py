import os
from random import uniform

from experiment import Experiment
from TicTacToe.experiments.trainPGStrategySupervised import TrainPGStrategySupervised
from TicTacToe.experiments.trainPGSupervisedContinuous import TrainPGSupervisedContinuous


class SupervisedCrossValidation(Experiment):

    def __init__(self, nested_experiment):
        super(SupervisedCrossValidation, self).__init__(os.path.dirname(os.path.abspath(__file__)))
        self.nested_experiment = nested_experiment()

    def run(self, iterations, lr_lower_boundary, lr_upper_boundary):

        GAMES = 10
        EPISODES = 1000

        for i in range(iterations):
            LR = 10 ** uniform(lr_lower_boundary, lr_upper_boundary)

            print("\nIteration %s/% - lr: %s" % (i+1, iterations, LR))

            self.nested_experiment.run(games=GAMES, episodes=EPISODES, lr=LR)
            self.nested_experiment.plot_and_save(file_name="TrainReinforcePlayerWithSharedNetwork lr: %s" % LR, plot_title="Lr: %s - %s Games - %s episodes" % (LR, GAMES, EPISODES))


if __name__ == '__main__':

    AVAILABLE_EXPERIMENTS = TrainPGStrategySupervised, TrainPGSupervisedContinuous

    experiment = SupervisedCrossValidation(TrainPGStrategySupervised)
    experiment.run(5, -4, -5.5)
