from datetime import datetime
from random import random, uniform, randint
import numpy as np

import TicTacToe.config as config
from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from TicTacToe.players.acPlayer import SmallFCACPlayer, FCACPlayer, ConvACPlayer
from TicTacToe.players.basePlayers import ExperiencedPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.evaluation import evaluate_against_base_players, evaluate_both_players, evaluate_against_each_other
from plotting import Printer


class TrainACPlayerVsBest(TicTacToeBaseExperiment):

    def __init__(self, games, evaluations, pretrained_player=None):
        super(TrainACPlayerVsBest, self).__init__()
        self.games = games
        self.evaluations = evaluations
        self.pretrained_player = pretrained_player.copy(shared_weights=False) if pretrained_player else None

        self.__plotter__.line3_name = "ExperiencedPlayer score"

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations, pretrained_player=self.pretrained_player)
        return self

    def run(self, lr, batch_size, silent=False):
        self.player1 = self.pretrained_player if self.pretrained_player else ConvACPlayer(lr=lr, batch_size=batch_size)

        # Player 2 has the same start conditions as Player 1 but does not train
        self.player2 = self.player1.copy(shared_weights=False)
        self.player2.strategy.train = False

        self.simulation = TicTacToe([self.player1, self.player2])

        games_per_evaluation = self.games // self.evaluations
        self.replacements = []
        start_time = datetime.now()
        for episode in range(1, self.evaluations+1):
            # train
            self.player1.strategy.train, self.player1.strategy.model.training = True, True  # training mode

            results, losses = self.simulation.run_simulations(games_per_evaluation)
            self.add_results(("Losses", np.mean(losses)))

            # evaluate
            self.player1.strategy.train, self.player1.strategy.model.training = False, False  # eval mode
            score, results, overview = evaluate_against_base_players(self.player1)
            self.add_results(results)

            if not silent:
                if Printer.print_episode(episode*games_per_evaluation, self.games, datetime.now() - start_time):
                    self.plot_and_save(
                        "%s vs BEST LR: %s" % (COMMENT, lr),
                        "%s Train %s vs Best version of self\nLR: %s Games: %s\nTime: %s"
                        % (COMMENT, self.player1, lr, episode*games_per_evaluation, config.time_diff(start_time)))

            if evaluate_against_each_other(self.player1, self.player2):
            # if evaluate_both_players(self.player1, self.player2):
                self.player2 = self.player1.copy(shared_weights=False)
                self.player2.strategy.train = False
                self.replacements.append(episode)

        print("Best player replaced after episodes: %s" % self.replacements)
        self.final_score, self.final_results, self.results_overview = evaluate_against_base_players(self.player1, silent=False)
        return self


if __name__ == '__main__':

    COMMENT = "BOOTSTRAP"
    ITERATIONS = 2

    start = datetime.now()
    for i in range(ITERATIONS):

        print("|| ITERATION: %s/%s ||" % (i+1, ITERATIONS))
        GAMES = 1000000
        EVALUATIONS = 1000  # 100 * randint(10, 500)
        LR = uniform(6e-4, 1e-5)  # random()*1e-9 + 1e-5
        BATCH_SIZE = 1

        PLAYER = None  # Experiment.load_player("Pretrain player [all traditional opponents].pth")

        experiment = TrainACPlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER)
        experiment.run(lr=LR, batch_size=BATCH_SIZE)

        print("\nSuccessfully trained on %s games\n" % experiment.num_episodes)
        if PLAYER:
            print("Pretrained on %s legal moves" % 1000000)
    print("Experiment completed successfully, Time: %s" % (datetime.now()-start))
