import os
from datetime import datetime

import TicTacToe.config as config
from experiment import Experiment
from TicTacToe.players.reinforcePlayer import PGStrategy
from TicTacToe.players.base_players import ExperiencedPlayer, RandomPlayer
from TicTacToe.environment.board import TicTacToeBoard
from plotting import Printer


class TrainPGSupervisedContinuous(Experiment):

    def __init__(self, games):
        super(TrainPGSupervisedContinuous, self).__init__(os.path.dirname(os.path.abspath(__file__)))

        self.games = games

    def reset(self):
        self.__init__(games=self.games)
        return self

    def run(self, lr, silent=False):

        strategy = PGStrategy(lr=lr)
        expert = ExperiencedPlayer(deterministic=True, block_mid=True)
        expert.color = config.BLACK

        generator = RandomPlayer()
        color_iterator = Experiment.AlternatingColorIterator()

        start = datetime.now()
        for game in range(self.games):
            acc_reward = 0
            acc_loss = 0
            board = TicTacToeBoard()

            for i in range(9):
                expert_move = expert.get_move(board)
                strategy_move = strategy.evaluate(board.board)

                reward = config.BLACK if expert_move == strategy_move else config.WHITE
                acc_loss += strategy.update(reward)
                acc_reward += reward

                # prepare for next sample
                move = generator.get_move(board)
                board.apply_move(move, color_iterator.__next__())

            self.add_losses([acc_loss / 9])
            self.add_scores(acc_reward / 9)

            if not silent:
                if Printer.print_episode(game + 1, self.games, datetime.now() - start):
                    self.plot_and_save("TrainReinforcePlayerWithSharedNetwork lr: %s" % lr, "Lr: %s - %s Games - Final reward: %s" % (lr, game+1, acc_reward))

        return acc_reward/9


if __name__ == '__main__':

    GAMES = 100000
    LR = 2e-5

    experiment = TrainPGSupervisedContinuous(games=GAMES)
    reward = experiment.run(lr=LR)

    print("Successively trained on %s games" % experiment.__plotter__.num_episodes)
