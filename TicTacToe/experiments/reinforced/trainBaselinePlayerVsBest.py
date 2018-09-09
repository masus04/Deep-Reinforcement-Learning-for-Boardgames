from datetime import datetime
from random import random, uniform, randint
import numpy as np

import TicTacToe.config as config
from TicTacToe.experiments.ticTacToeBaseExperiment import TicTacToeBaseExperiment
from TicTacToe.players.baselinePlayer import FCBaseLinePlayer, LargeFCBaseLinePlayer
from TicTacToe.players.basePlayers import ExperiencedPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.evaluation import evaluate_against_base_players, evaluate_both_players, evaluate_against_each_other
from plotting import Printer


class TrainBaselinePlayerVsBest(TicTacToeBaseExperiment):

    def __init__(self, games, evaluations, pretrained_player=None):
        super(TrainBaselinePlayerVsBest, self).__init__()
        self.games = games
        self.evaluations = evaluations
        self.pretrained_player = pretrained_player.copy(shared_weights=False) if pretrained_player else None

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations, pretrained_player=self.pretrained_player)
        return self

    def run(self, lr, weight_decay, silent=False):
        self.player1 = self.pretrained_player if self.pretrained_player else FCBaseLinePlayer(lr=lr, weight_decay=weight_decay)

        # Player 2 has the same start conditions as Player 1 but does not train
        self.player2 = self.player1.copy(shared_weights=False)
        self.player2.strategy.train = False

        games_per_evaluation = self.games // self.evaluations
        self.replacements = []
        start_time = datetime.now()
        for episode in range(1, self.evaluations+1):
            # train
            self.player1.strategy.train, self.player1.strategy.model.training = True, True  # training mode

            self.simulation = TicTacToe([self.player1, self.player2])
            results, losses = self.simulation.run_simulations(games_per_evaluation)
            self.add_loss(np.mean(losses))
            self.add_results(("Loss", np.mean(losses)))
            self.add_results(("Best", np.mean(results)))

            # evaluate
            if episode*games_per_evaluation % 1000 == 0:
                self.player1.strategy.train, self.player1.strategy.model.training = False, False  # eval mode
                score, results, overview = evaluate_against_base_players(self.player1)
                self.add_results(results)

                if not silent and Printer.print_episode(episode*games_per_evaluation, self.games, datetime.now() - start_time):
                    self.plot_and_save(
                        "%s vs BEST" % (self.player1),
                        "Train %s vs Best version of self\nGames: %s Evaluations: %s\nTime: %s"
                        % (self.player1, episode*games_per_evaluation, self.evaluations, config.time_diff(start_time)))

            if evaluate_against_each_other(self.player1, self.player2):
                self.player2 = self.player1.copy(shared_weights=False)
                self.player2.strategy.train = False
                self.replacements.append(episode)

        print("Best player replaced after episodes: %s" % self.replacements)
        self.final_score, self.final_results, self.results_overview = evaluate_against_base_players(self.player1, silent=False)
        return self


if __name__ == '__main__':

    ITERATIONS = 1

    start = datetime.now()
    for i in range(ITERATIONS):

        print("|| ITERATION: %s/%s ||" % (i+1, ITERATIONS))
        GAMES = 1000000
        EVALUATIONS = GAMES//100  # 100 * randint(10, 500)
        LR = random()*1e-9 + 1e-3  # uniform(1e-4, 2e-5)  # random()*1e-9 + 1e-5
        WEIGHT_DECAY = 0.01

        PLAYER = None  # Experiment.load_player("Pretrain player [all traditional opponents].pth")

        experiment = TrainBaselinePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER)
        experiment.run(lr=LR, weight_decay=WEIGHT_DECAY)
        experiment.save_player(experiment.player1)

        print("\nSuccessfully trained on %s games\n" % experiment.num_episodes)
        if PLAYER:
            print("Pretrained on %s legal moves" % 1000000)
    print("Experiment completed successfully, Time: %s" % (datetime.now()-start))
