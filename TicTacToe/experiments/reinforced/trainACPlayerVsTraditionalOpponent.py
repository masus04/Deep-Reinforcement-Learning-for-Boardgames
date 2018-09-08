from datetime import datetime
from random import random, choice, uniform
import numpy as np

import TicTacToe.config as config
from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from TicTacToe.players.basePlayers import RandomPlayer, ExperiencedPlayer, ExpertPlayer
from TicTacToe.players.acPlayer import FCACPlayer, LargeFCACPlayer, ConvACPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.evaluation import evaluate_against_base_players, format_overview
from plotting import Printer


class TrainACPlayerVsTraditionalOpponent(TicTacToeBaseExperiment):

    def __init__(self, games, evaluations, pretrained_player, opponent):
        super(TrainACPlayerVsTraditionalOpponent, self).__init__()
        self.games = games
        self.evaluations = evaluations
        self.pretrained_player = pretrained_player.copy(shared_weights=False) if pretrained_player else None
        self.opponent = opponent
        self.milestones = []

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations, pretrained_player=self.pretrained_player, opponent=self.opponent)
        return self

    def run(self, lr, silent=False):

        self.player1 = self.pretrained_player if self.pretrained_player else FCACPlayer(lr=lr)

        if self.opponent is not None:
            self.player2 = self.opponent
            self.simulation = TicTacToe([self.player1, self.player2])

        games_per_evaluation = self.games // self.evaluations
        start_time = datetime.now()
        for episode in range(1, self.evaluations+1):

            if self.opponent is None:
                self.player2 = choice((RandomPlayer(), ExpertPlayer(), ExperiencedPlayer(deterministic=False, block_mid=True)))
                self.simulation = TicTacToe([self.player1, self.player2])

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
                    overview = format_overview(overview)
                    self.plot_and_save(
                        "ReinforcementTraining vs %s LR: %s" % (self.opponent, lr),
                        "Train %s vs traditional opponents: %s \nLR: %s Games: %s\ntook: %s"
                        % (self.player1, self.opponent, lr, episode*games_per_evaluation, config.time_diff(start_time)))

        self.final_score, self.final_results, self.results_overview = evaluate_against_base_players(self.player1, silent=False)
        return self


if __name__ == '__main__':

    MILESTONES = True
    start = datetime.now()

    for i in range(15):
        GAMES = 50000
        EVALUATIONS = 5000
        LR = random()*1e-15 + 1e-3  # uniform(1e-2, 1e-4)

        PLAYER = None  # Experiment.load_player("ReinforcePlayer using 3 layers pretrained on legal moves for 1000000 games.pth")
        OPPONENT = None  # ExperiencedPlayer(deterministic=False, block_mid=False)

        print("Training ACPlayer vs %s with lr: %s" % (OPPONENT, LR))
        experiment = TrainACPlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=OPPONENT)
        experiment.run(lr=LR)
        print()
        experiment.save_player(experiment.player1, "%s pretrained on traditional opponents" % experiment.player1)

        print("Successfully trained on %s games, pretrained on %s" % (experiment.__plotter__.num_episodes, 10000000))

    print("took: %s" % (datetime.now() - start))
