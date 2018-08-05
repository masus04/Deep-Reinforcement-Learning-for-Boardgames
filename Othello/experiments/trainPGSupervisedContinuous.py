import os
from datetime import datetime
from random import random

import Othello.config as config
from Othello.experiments.othelloBaseExperiment import OthelloBaseExperiment
from Othello.players.acPlayer import FCACPlayer, ConvACPlayer
from Othello.players.reinforcePlayer import FCReinforcePlayer, FCReinforcePlayer, HugeFCReinforcePlayer, ConvReinforcePlayer
from Othello.players.basePlayers import ExperiencedPlayer, RandomPlayer
from Othello.environment.board import OthelloBoard
from plotting import Printer


class TrainPGSupervisedContinuous(OthelloBaseExperiment):

    def __init__(self, games, evaluation_period):
        super(TrainPGSupervisedContinuous, self).__init__()

        self.games = games
        self.evaluation_period = evaluation_period

    def reset(self):
        self.__init__(games=self.games, evaluation_period=self.evaluation_period)
        return self

    def run(self, lr, player, silent=False):

        EVALUATION_GAMES = 10

        player.color = config.BLACK

        expert = ExperiencedPlayer(deterministic=True)
        expert.color = config.BLACK

        generator = RandomPlayer()
        color_iterator = self.AlternatingColorIterator()

        validation_set = self.generate_supervised_training_data(EVALUATION_GAMES, ExperiencedPlayer(deterministic=True))

        print("Training ReinforcedPlayer supervised continuously with LR: %s" % lr)
        start = datetime.now()
        for game in range(self.games):
            rewards = []
            board = OthelloBoard()

            while len(board.get_valid_moves(expert.color)) != 0:
                expert_move = expert.get_move(board)
                player_move = player.get_move(board)

                reward = config.LABEL_WIN if expert_move == player_move else config.LABEL_LOSS
                rewards.append(reward)

                # prepare for next sample
                color = color_iterator.__next__()
                generator.color = color
                move = generator.get_move(board)
                if move is not None:
                    board.apply_move(move, color)

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

    ITERATIONS = 5

    for i in range(ITERATIONS):
        GAMES = 5000000
        LR = random()*1e-9 + 1e-4

        EVALUATION_PERIOD = 100
        PLAYER = FCReinforcePlayer(lr=LR)

        experiment = TrainPGSupervisedContinuous(games=GAMES, evaluation_period=EVALUATION_PERIOD)
        reward = experiment.run(player=PLAYER, lr=LR)

        print("Successfully trained on %s games" % experiment.__plotter__.num_episodes)
