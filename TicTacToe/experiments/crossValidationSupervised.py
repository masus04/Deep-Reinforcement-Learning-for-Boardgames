import os
from random import uniform

from experiment import Experiment
from TicTacToe.experiments.trainPGStrategySupervised import TrainPGStrategySupervised


class SupervisedCrossValidation(Experiment):

    def __init__(self):
        super(SupervisedCrossValidation, self).__init__(os.path.dirname(os.path.abspath(__file__)))

    def run(self, iterations, lr_boundaries):

        GAMES = 1
        EPISODES = 100000

        nested_experiment = TrainPGStrategySupervised()

        for i in range(iterations):
            LR = 10 ** uniform(lr_boundaries[0], lr_boundaries[1])

            print("\nIteration %s/% - lr: %s" % (i+1, iterations, LR))

            nested_experiment.run(games=GAMES, episodes=EPISODES, lr=LR)
            nested_experiment.plot_and_save(file_name="TrainReinforcePlayerWithSharedNetwork lr: %s" % LR, plot_title="Lr: %s - %s Games - %s episodes" % (LR, GAMES, EPISODES))


if __name__ == '__main__':
    experiment = SupervisedCrossValidation()
    experiment.run(5, (-4.5, -6))
