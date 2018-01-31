import os
from datetime import datetime

import TicTacToe.config as config
from experiment import Experiment
from TicTacToe.players.reinforcePlayer import PGStrategy
from TicTacToe.players.base_players import ExperiencedPlayer
from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from plotting import Printer


class TrainPGStrategySupervised(TicTacToeBaseExperiment):

    def __init__(self, games, episodes):
        super(TrainPGStrategySupervised, self).__init__(os.path.dirname(os.path.abspath(__file__)))

        self.games = games
        self.episodes = episodes

    def reset(self):
        self.__init__(games=self.games, episodes=self.episodes)
        return self

    def run(self, lr, silent=False):

        TEST_GAMES = 1

        strategy = PGStrategy(lr=lr)
        expert = ExperiencedPlayer(deterministic=True, block_mid=True)
        expert.color = config.BLACK

        training_set = self.generate_supervised_training_data(self.games, expert)
        test_set = self.generate_supervised_training_data(TEST_GAMES, expert)

        start = datetime.now()
        for episode in range(self.episodes):
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
                test_reward = config.BLACK if expert_move == strategy_move else config.WHITE

                acc_test_reward += test_reward

            average_reward = acc_reward/len(training_set)
            average_test_reward = acc_test_reward/len(test_set)

            self.add_scores(average_reward, average_test_reward)
            self.add_losses([acc_loss/len(training_set)])

            if not silent:
                if Printer.print_episode(episode + 1, self.episodes, datetime.now() - start):
                    self.plot_and_save("TrainReinforcePlayerWithSharedNetwork lr: %s" % lr, "Lr: %s - %s Games - %s Episodes\nFinal Scores: %s / %s" %
                                       (lr, self.games, episode+1, '{:.2f}'.format(average_reward), '{:.2f}'.format(average_test_reward)))

        return average_reward, average_test_reward


if __name__ == '__main__':

    GAMES = 2
    EPISODES = 40000 // GAMES
    LR = 2e-5

    experiment = TrainPGStrategySupervised(games=GAMES, episodes=EPISODES)
    experiment.run(lr=LR)

    print("Successively trained on %s games" % experiment.__plotter__.num_episodes)
