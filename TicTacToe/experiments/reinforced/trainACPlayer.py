from datetime import datetime
from random import random
import numpy as np

from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from TicTacToe.players.acPlayer import FCACPlayer, LargeFCACPlayer, ConvACPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.evaluation import evaluate_against_base_players
from plotting import Printer


class TrainACPlayer(TicTacToeBaseExperiment):

    def __init__(self, games, evaluations, pretrained_player=None):
        super(TrainACPlayer, self).__init__()
        self.games = games
        self.evaluations = evaluations
        self.pretrained_player = pretrained_player.copy(shared_weights=False) if pretrained_player else None

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations, pretrained_player=self.pretrained_player)
        return self

    def run(self, lr, silent=False):

        self.player1 = self.pretrained_player if self.pretrained_player else FCACPlayer(lr=lr)

        # Player2 shares the same weights but does not change them.
        self.player2 = self.player1.copy(shared_weights=True)
        self.player2.strategy.train = False

        self.simulation = TicTacToe([self.player1, self.player2])

        games_per_evaluation = self.games // self.evaluations
        start_time = datetime.now()
        for episode in range(1, self.evaluations+1):
            # train
            self.player1.strategy.train, self.player1.strategy.model.training = True, True  # training mode

            results, losses = self.simulation.run_simulations(games_per_evaluation)
            self.add_results(("Losses", np.mean(losses)))

            # evaluate  # TODO: Test this
            if episode % 1000 == 0:
                self.player1.strategy.train, self.player1.strategy.model.training = False, False  # eval mode
                score, results, overview = evaluate_against_base_players(self.player1)
                self.add_results(results)

                if not silent:
                    if Printer.print_episode(episode*games_per_evaluation, self.games, datetime.now() - start_time):
                        self.plot_and_save(
                            "ReinforcementTraining LR: %s" % lr,
                            "Train ACPlayer vs self with shared network\nLR: %s Games: %s" % (lr, episode*games_per_evaluation))

        self.final_score, self.final_results, self.results_overview = evaluate_against_base_players(self.player1, silent=False)
        return self


if __name__ == '__main__':

    GAMES = 100000
    EVALUATIONS = 1000
    LR = random() * 1e-9 + 1e-5

    PLAYER = None  # Experiment.load_player("FCACPlayer pretrained on traditional opponents.pth")

    print("Training ACPlayer vs self with lr: %s" % LR)
    experiment = TrainACPlayer(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER)
    experiment.run(lr=LR)

    print("\nSuccessfully trained on %s games" % experiment.num_episodes)
    if PLAYER:
        print("Pretrained on %s legal moves" % 1000000)
