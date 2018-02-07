import os
from datetime import datetime
from random import random

import TicTacToe.config as config
from experiment import Experiment
from TicTacToe.players.reinforcePlayer import ReinforcePlayer, PGStrategy
from TicTacToe.players.base_players import ExperiencedPlayer
from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from plotting import Printer


class TrainPGStrategySupervised(TicTacToeBaseExperiment):

    def __init__(self, games, episodes):
        super(TrainPGStrategySupervised, self).__init__()

        self.games = games
        self.episodes = episodes

    def reset(self):
        self.__init__(games=self.games, episodes=self.episodes)
        return self

    def run(self, lr, silent=False):

        print("Training PGStrategy supervised on %s games for %s Episodes - LR: %s" % (self.games, self.episodes, lr))
        TEST_GAMES = 1

        player = ReinforcePlayer(strategy=PGStrategy, lr=lr)
        player.color = config.BLACK

        expert = ExperiencedPlayer(deterministic=True, block_mid=True)
        expert.color = config.BLACK

        training_set = self.generate_supervised_training_data(self.games, expert)
        test_set = self.generate_supervised_training_data(TEST_GAMES, expert)

        start = datetime.now()
        for episode in range(self.episodes):
            rewards = []
            test_rewards = []

            for board, expert_move in training_set:
                # Training mode
                player.strategy.train, player.strategy.model.training = True, True

                strategy_move = player.get_move(board)
                reward = config.BLACK if expert_move == strategy_move else config.WHITE
                rewards.append(reward)

            # Player.register_winner stand in because it does not allow more than one label
            for reward in rewards:
                player.strategy.rewards.append(reward)
            player.num_moves = 0
            loss = player.strategy.update()

            for board, expert_move in test_set:
                # Evaluation mode
                player.strategy.train, player.strategy.model.training = False, False

                strategy_move = player.get_move(board)
                test_reward = config.BLACK if expert_move == strategy_move else config.WHITE
                test_rewards.append(test_reward)

            average_reward = sum(rewards)/len(rewards)
            average_test_reward = sum(test_rewards)/len(test_rewards)

            self.add_scores(average_reward, average_test_reward)
            self.add_loss(loss)

            if not silent:
                if Printer.print_episode(episode + 1, self.episodes, datetime.now() - start):
                    plot_name = "Supervised on %s games lr: %s" % (self.games, lr)
                    plot_info = "Lr: %s - %s Games - %s Episodes\nFinal Scores: %s / %s \nTime: %s" % (lr, self.games, episode+1, '{:.2f}'.format(average_reward), '{:.2f}'.format(average_test_reward), config.time_diff(start))
                    self.plot_and_save(plot_name, plot_name + "\n" + plot_info)

        return average_reward, average_test_reward


if __name__ == '__main__':

    GAMES = 1
    EPISODES = 10000
    LR = random()*1e-9 + 2e-5

    experiment = TrainPGStrategySupervised(games=GAMES, episodes=EPISODES)
    experiment.run(lr=LR)

    print("Successfully trained on %s games" % experiment.__plotter__.num_episodes)
