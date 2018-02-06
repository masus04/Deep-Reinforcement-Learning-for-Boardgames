import os
from datetime import datetime
from random import random
from torch.multiprocessing import Pool
import numpy as np

import TicTacToe.config as config
from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from TicTacToe.players.reinforcePlayer import PGStrategy, ReinforcePlayer
from TicTacToe.players.base_players import ExperiencedPlayer, RandomPlayer
from TicTacToe.environment.board import TicTacToeBoard
from plotting import Printer


class TrainPGSupervisedContinuous(TicTacToeBaseExperiment):

    def __init__(self, batches, evaluation_period):
        super(TrainPGSupervisedContinuous, self).__init__(os.path.dirname(os.path.abspath(__file__)))

        self.batches = batches
        self.evaluation_period = evaluation_period

    def reset(self):
        self.__init__(batches=self.batches, evaluation_period=self.evaluation_period)
        return self

    def run(self, lr, batch_size=1, silent=False):

        EVALUATION_GAMES = 10

        player = ReinforcePlayer(strategy=PGStrategy, lr=lr, batch_size=batch_size)
        player.color = config.BLACK

        validation_set = self.generate_supervised_training_data(EVALUATION_GAMES, ExperiencedPlayer(deterministic=True, block_mid=True))

        print("Training ReinforcedPlayer supervised continuously with LR: %s" % lr)
        start = datetime.now()

        WORKERS = 4
        pool = Pool(processes=WORKERS)
        player.share_memory()
        for batch in range(1, self.batches+1):
            results = pool.map_async(self.multiprocessing_train, [player.copy() for i in range(batch_size)]).get()

            batch_reward = 0
            for r in results:
                self.add_loss(r[0])
                self.add_scores(r[1])
                batch_reward += r[1]

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
            self.add_scores(None, average_test_reward)

            if not silent:
                if Printer.print_episode(batch, self.batches, datetime.now() - start):
                    plot_name = "Supervised Continuous lr: %s batch size: %s" % (lr, batch_size)
                    plot_info = "%s Batches - Final reward: %s \nTime: %s" % (batch, batch_reward, config.time_diff(start))
                    self.plot_and_save(plot_name, plot_name + "\n" + plot_info)

        pool.close()
        return batch_reward

    def multiprocessing_train(self, player):
        player = player.copy(shared_weights=True)

        expert = ExperiencedPlayer(deterministic=True, block_mid=True)
        expert.color = config.BLACK

        generator = RandomPlayer()
        color_iterator = self.AlternatingColorIterator()

        board = TicTacToeBoard()
        rewards = []
        for i in range(9):
            expert_move = expert.get_move(board)
            player_move = player.get_move(board)

            reward = config.BLACK if expert_move == player_move else config.WHITE
            rewards.append(reward)

            # prepare for next sample
            move = generator.get_move(board)
            board.apply_move(move, color_iterator.__next__())

        # Player.register_winner stand in because it does not allow more than one label
        for reward in rewards:
            player.strategy.rewards.append(reward)
        player.num_moves = 0
        loss = player.strategy.update()

        return loss, sum(rewards) / 9


if __name__ == '__main__':

    BATCHES = 2
    BATCH_SIZE = 32
    LR = random()*1e-9 + 1e-3

    EVALUATION_PERIOD = 100

    experiment = TrainPGSupervisedContinuous(batches=BATCH_SIZE, evaluation_period=EVALUATION_PERIOD)
    reward = experiment.run(lr=LR, batch_size=BATCH_SIZE)

    print("Successfully trained on %s games" % experiment.__plotter__.num_episodes)
