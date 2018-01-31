import os
from datetime import datetime

import TicTacToe.config as config
from experiment import Experiment
from TicTacToe.players.reinforcePlayer import PGStrategy
from TicTacToe.players.base_players import ExperiencedPlayer
from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from plotting import Printer


class TrainPGStrategySupervised(TicTacToeBaseExperiment):

    def __init__(self):
        super(TrainPGStrategySupervised, self).__init__(os.path.dirname(os.path.abspath(__file__)))

    def run(self, games, test_games, episodes, lr):

        strategy = PGStrategy(lr=lr)
        expert = ExperiencedPlayer(deterministic=True, block_mid=True)
        expert.color = config.BLACK

        training_set = self.generate_supervised_training_data(games, expert)
        test_set = self.generate_supervised_training_data(test_games, expert)

        start = datetime.now()
        for game in range(episodes):
            acc_reward = 0
            acc_test_reward = 0
            acc_loss = 0

            for sample, expert_move in training_set:
                strategy_move = strategy.evaluate(sample)
                reward = config.BLACK if expert_move == strategy_move else config.WHITE
                loss = strategy.update(reward)

                acc_reward += reward
                acc_loss += loss

            for sample, expert_move in test_set:
                strategy_move = strategy.evaluate(sample)
                acc_test_reward = config.BLACK if expert_move == strategy_move else config.WHITE

                acc_reward += acc_test_reward

            self.add_scores(acc_reward/len(training_set), acc_test_reward/len(test_set))
            self.add_losses([acc_loss/len(training_set)])

            if (game+1) % 500 == 0:
                self.plot_and_save("TrainReinforcePlayerWithSharedNetwork lr: %s" % lr, "Lr: %s - %s Games" % (lr, game+1))

            Printer.print_episode(game+1, episodes, datetime.now() - start)


if __name__ == '__main__':

    GAMES = 20
    TEST_GAMES = 10
    EPISODES = 1000
    LR = 1e-5

    experiment = TrainPGStrategySupervised()
    experiment.run(games=GAMES, test_games=TEST_GAMES, episodes=EPISODES, lr=LR)
    experiment.plot_and_save("TrainReinforcePlayerWithSharedNetwork lr: %s" % LR, "Lr: %s - %s Games" % (LR, GAMES))

    print("Successively trained on %s games" % experiment.__plotter__.num_episodes)
