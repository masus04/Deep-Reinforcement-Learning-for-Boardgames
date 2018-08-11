from datetime import datetime
from random import random, choice, uniform
import numpy as np

import Othello.config as config
from Othello.experiments.othelloBaseExperiment import OthelloBaseExperiment
from Othello.players.basePlayers import RandomPlayer, DeterministicPlayer, NovicePlayer, ExperiencedPlayer
from Othello.players.acPlayer import FCACPlayer, LargeFCACPlayer, HugeFCACPlayer, ConvACPlayer
from Othello.environment.game import Othello
from Othello.environment.evaluation import evaluate_against_base_players, format_overview
from plotting import Printer


class TrainACPlayerVsTraditionalOpponent(OthelloBaseExperiment):

    def __init__(self, games, evaluations, pretrained_player, opponent):
        super(TrainACPlayerVsTraditionalOpponent, self).__init__()
        self.games = games
        self.evaluations = evaluations
        self.pretrained_player = pretrained_player.copy(shared_weights=False) if pretrained_player else None
        self.opponent = opponent

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations, pretrained_player=self.pretrained_player, opponent=self.opponent)
        return self

    def run(self, lr, silent=False):
        self.player1 = self.pretrained_player if self.pretrained_player else FCACPlayer(lr=lr)  # ConvACPlayer(lr=lr)

        if self.opponent is not None:
            self.player2 = self.opponent
            self.simulation = Othello([self.player1, self.player2])

        games_per_evaluation = self.games // self.evaluations
        start_time = datetime.now()
        for episode in range(1, self.evaluations+1):

            if self.opponent is None:
                self.player2 = choice((RandomPlayer(), NovicePlayer(), ExperiencedPlayer(deterministic=False)))
                self.simulation = Othello([self.player1, self.player2])

            # train
            self.player1.strategy.train, self.player1.strategy.model.training = True, True  # training mode

            results, losses = self.simulation.run_simulations(games_per_evaluation)
            self.add_results(("Losses", np.mean(losses)))

            # evaluate
            self.player1.strategy.train, self.player1.strategy.model.training = False, False  # eval mode
            if self.opponent is None:
                score, results, overview = evaluate_against_base_players(self.player1)
            else:
                score, results, overview = evaluate_against_base_players(self.player1, evaluation_players=[self.opponent])

            self.add_results(results)

            if not silent:
                if Printer.print_episode(episode*games_per_evaluation, self.games, datetime.now() - start_time):
                    overview = format_overview(overview)
                    self.plot_and_save(
                        "%s vs TRADITIONAL OPPONENT" % (self.player1),
                        "Train %s vs %s\nGames: %s Evaluations: %s\nTime: %s"
                        % (self.player1, self.opponent, episode*games_per_evaluation, self.evaluations, config.time_diff(start_time)))

        self.final_score, self.final_results, self.results_overview = evaluate_against_base_players(self.player1, silent=False)
        return self


if __name__ == '__main__':

    ITERATIONS = 1
    start = datetime.now()

    for i in range(ITERATIONS):
        print("Iteration %s/%s" % (i + 1, ITERATIONS))
        GAMES = 2000000
        EVALUATIONS = GAMES//1000
        LR = random()*1e-9 + 5e-5  # uniform(1e-2, 1e-4)

        PLAYER = None  # Experiment.load_player("ReinforcePlayer using 3 layers pretrained on legal moves for 1000000 games.pth")
        OPPONENT = None  # ExperiencedPlayer(deterministic=True)

        print("Training ReinforcePlayer vs %s with lr: %s" % (OPPONENT, LR))
        experiment = TrainACPlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=OPPONENT)
        experiment.run(lr=LR)
        experiment.save_player(experiment.player1)
        
    print("Successfully trained on %s games, pretrained on %s" % (experiment.__plotter__.num_episodes, 10000000))

    print("took: %s" % (datetime.now() - start))
