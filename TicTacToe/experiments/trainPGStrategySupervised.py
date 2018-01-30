import os
from datetime import datetime

import TicTacToe.config as config
from experiment import Experiment
from TicTacToe.players.reinforcePlayer import PGStrategy
from TicTacToe.players.base_players import ExperiencedPlayer, RandomPlayer
from TicTacToe.environment.board import TicTacToeBoard
from plotting import Printer


class TrainPGStrategySupervised(Experiment):

    def __init__(self):
        super(TrainPGStrategySupervised, self).__init__(os.path.dirname(os.path.abspath(__file__)))

    def run(self, games, episodes, lr):

        strategy = PGStrategy(lr=lr)
        expert = ExperiencedPlayer(deterministic=True, block_mid=True)
        expert.color = config.BLACK

        generator = RandomPlayer()
        color_iterator = AlternatingColorIterator()

        start = datetime.now()
        training_set = []
        for game in range(games):
            board = TicTacToeBoard()
            for i in range(9):
                # generate training pair
                expert_move = expert.get_move(board)
                training_set.append((board.board.copy(), expert_move))

                # prepare for next sample
                move = generator.get_move(board)
                board.apply_move(move, color_iterator.__next__())

        print("Generated %s training pairs form %s games in %s" % (len(training_set), games, datetime.now()-start))

        start = datetime.now()
        acc_reward = 0
        for i in range(episodes):
            for sample, expert_move in training_set:
                strategy_move = strategy.evaluate(sample)
                reward = config.BLACK if expert_move == strategy_move else config.WHITE
                loss = strategy.update(reward)

                acc_reward += reward
                self.add_losses([loss])

            self.add_score(acc_reward/len(training_set))
            acc_reward = 0

            Printer.print_episode(i+1, episodes, datetime.now() - start)


class AlternatingColorIterator:
    def __init__(self):
        self.colors = [config.BLACK, config.WHITE]

    def __iter__(self):
        return self

    def __next__(self):
        self.colors = list(reversed(self.colors))
        return self.colors[-1]


if __name__ == '__main__':

    GAMES = 20
    EPISODES = 500
    LR = 1e-4

    experiment = TrainPGStrategySupervised()
    experiment.run(games=GAMES, episodes=EPISODES, lr=LR)
    experiment.plot_and_save("TrainReinforcePlayerWithSharedNetwork lr: %s" % LR)

    print("Successively trained on %s games" % experiment.__plotter__.num_episodes)
