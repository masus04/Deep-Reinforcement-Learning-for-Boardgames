import os
from datetime import datetime

from experiment import Experiment
from TicTacToe.players.reinforcePlayer import ReinforcePlayer, Strategy
from TicTacToe.environment.game import TicTacToe
from TicTacToe.players.base_players import evaluate_against_base_players
from plotting import Printer


class TrainReinforcePlayer(Experiment):

    def __init__(self, games, evaluations):
        super(TrainReinforcePlayer, self).__init__(os.path.dirname(os.path.abspath(__file__)))

        self.player1 = ReinforcePlayer(Strategy, lr=0.001)

        # Player2 shares the same weights but does not change them.
        self.player2 = self.player1.copy(shared_weights=True)
        self.player2.strategy.train = False

        self.simulation = TicTacToe([self.player1, self.player2])

        self.games = games
        self.evaluations = evaluations

    def run(self):
        start_time = datetime.now()

        for episode in range(self.evaluations):
            # train
            self.player1.strategy.train = True
            results, losses = self.simulation.run_simulations(self.games // self.evaluations)
            self.add_losses(losses)    # losses are interesting during training

            # evaluate
            self.player1.strategy.train = False
            result = evaluate_against_base_players(self.player1)
            self.add_score(result)    # only evaluation results are interesting

            Printer.print_episode(episode+1, self.evaluations, datetime.now() - start_time)

        self.plot_scores("TrainReinforcePlayerWithSharedNetwork")
        return self


if __name__ == '__main__':

    GAMES = 10000
    EVALUATIONS = 100

    experiment = TrainReinforcePlayer(GAMES, EVALUATIONS)
    experiment.run()
    experiment.plot_and_save("TrainReinforcePlayerWithSharedNetwork")
    print("Successively trained on %s games" % experiment.__plotter__.num_episodes)

