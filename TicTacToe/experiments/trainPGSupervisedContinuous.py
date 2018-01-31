import os
from datetime import datetime

import TicTacToe.config as config
from experiment import Experiment, AlternatingColorIterator
from TicTacToe.players.reinforcePlayer import PGStrategy
from TicTacToe.players.base_players import ExperiencedPlayer, RandomPlayer
from TicTacToe.environment.board import TicTacToeBoard
from plotting import Printer

class TrainPGSupervisedContinuous(Experiment):

    def __init__(self):
        super(TrainPGSupervisedContinuous, self).__init__(os.path.dirname(os.path.abspath(__file__)))

    def run(self, games, lr):

        strategy = PGStrategy(lr=lr)
        expert = ExperiencedPlayer(deterministic=True, block_mid=True)
        expert.color = config.BLACK

        generator = RandomPlayer()
        color_iterator = AlternatingColorIterator()

        start = datetime.now()
        acc_reward = 0
        acc_loss = 0
        for game in range(games):
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
            acc_loss = 0
            acc_reward = 0
            Printer.print_episode(game + 1, games, datetime.now() - start)

            if (game+1) % 1000 == 0:
                self.plot_and_save("TrainReinforcePlayerWithSharedNetwork lr: %s" % lr, "Lr: %s - %s Games" % (lr, game+1))


if __name__ == '__main__':

    GAMES = 15000
    LR = 10**-5

    experiment = TrainPGSupervisedContinuous()
    experiment.run(games=GAMES, lr=LR)
    experiment.plot_and_save("TrainReinforcePlayerWithSharedNetwork lr: %s" % LR, "Lr: %s - %s Games" % (LR, GAMES))

    print("Successively trained on %s games" % experiment.__plotter__.num_episodes)
