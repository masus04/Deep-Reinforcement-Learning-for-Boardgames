import os
from datetime import datetime
from random import random

import TicTacToe.config as config
from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from TicTacToe.players.acPlayer import LargeFCACPlayer, ConvACPlayer
from TicTacToe.players.reinforcePlayer import FCReinforcePlayer, ConvReinforcePlayer
from TicTacToe.players.basePlayers import ExperiencedPlayer, RandomPlayer
from TicTacToe.environment.board import TicTacToeBoard
from plotting import Printer


class TrainPGSupervisedContinuous(TicTacToeBaseExperiment):

    def __init__(self, games, evaluation_period):
        super(TrainPGSupervisedContinuous, self).__init__()

        self.games = games
        self.evaluation_period = evaluation_period

    def reset(self):
        self.__init__(games=self.games, evaluation_period=self.evaluation_period)
        return self

    def run(self, lr, silent=False):

        EVALUATION_GAMES = 10

        player = FCReinforcePlayer(lr=lr)
        player.color = config.BLACK

        expert = ExperiencedPlayer(deterministic=True, block_mid=True)
        expert.color = config.BLACK

        generator = RandomPlayer()
        color_iterator = self.AlternatingColorIterator()

        validation_set = self.generate_supervised_training_data(EVALUATION_GAMES, ExperiencedPlayer(deterministic=True, block_mid=True))

        print("Training ReinforcedPlayer supervised continuously with LR: %s" % lr)
        start = datetime.now()
        for game in range(self.games):
            rewards = []
            board = TicTacToeBoard()

            for i in range(9):
                expert_move = expert.get_move(board)
                player_move = player.get_move(board)

                reward = config.LABEL_WIN if expert_move == player_move else config.LABEL_LOSS
                rewards.append(reward)

                # prepare for next sample
                move = generator.get_move(board)
                board.apply_move(move, color_iterator.__next__())

            average_reward = sum(rewards) / len(rewards)
            player.strategy.rewards = rewards
            loss = player.strategy.update()

            del rewards[:]
            self.add_results([("Losses", loss), ("Reward", average_reward)])

            if game % self.evaluation_period == 0:
                test_rewards = []
                for board, expert_move in validation_set:
                    # Evaluation mode
                    player.strategy.train, player.strategy.model.training = False, False
                    strategy_move = player.get_move(board)
                    player.strategy.train, player.strategy.model.training = True, True

                    test_reward = config.BLACK if expert_move == strategy_move else config.WHITE
                    test_rewards.append(test_reward)

                average_test_reward = sum(test_rewards) / len(test_rewards)
                del test_rewards[:]
                self.add_results(("Test reward", average_test_reward))

            if not silent:
                if Printer.print_episode(game + 1, self.games, datetime.now() - start):
                    plot_name = "Supervised Continuous training of %s" % (player)
                    plot_info = "%s Games - Final reward: %s \nTime: %s" % (game+1, average_reward, config.time_diff(start))
                    self.plot_and_save(plot_name, plot_name + "\n" + plot_info)

        return average_reward


if __name__ == '__main__':

    GAMES = 100000
    LR = random()*1e-9 + 1e-4

    EVALUATION_PERIOD = 1000

    experiment = TrainPGSupervisedContinuous(games=GAMES, evaluation_period=EVALUATION_PERIOD)
    reward = experiment.run(lr=LR)

    print("Successfully trained on %s games" % experiment.__plotter__.num_episodes)
