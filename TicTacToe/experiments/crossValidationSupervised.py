import os
from random import uniform
from datetime import datetime

from experiment import Experiment
from TicTacToe.experiments.trainPGStrategySupervised import TrainPGStrategySupervised
from TicTacToe.experiments.trainPGSupervisedContinuous import TrainPGSupervisedContinuous


class SupervisedCrossValidation(Experiment):

    def __init__(self, nested_experiment, batch_size):
        super(SupervisedCrossValidation, self).__init__(os.path.dirname(os.path.abspath(__file__)))
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

            last_reward = self.nested_experiment.reset().run(lr=LR, batch_size=self.batch_size)
            results.append((last_reward, LR))

        return sorted(results, reverse=True)


if __name__ == '__main__':

    AVAILABLE_EXPERIMENTS = TrainPGStrategySupervised, TrainPGSupervisedContinuous
    start = datetime.now()

    GAMES = 2
    EPISODES = 1000000
    BATCH_SIZE = 32

    # experiment = SupervisedCrossValidation(TrainPGStrategySupervised(games=GAMES, episodes=EPISODES))
    experiment = SupervisedCrossValidation(TrainPGSupervisedContinuous(games=EPISODES, evaluation_period=1000), batch_size=BATCH_SIZE)
    results = experiment.run(5, -3, -5)

    print("\nReward(s) - LR:")
    for res in results:
        print("%s - %s" % (res[0], res[1]))

    print("\nCrossvalidation complete, took: %s" % (datetime.now() - start))
