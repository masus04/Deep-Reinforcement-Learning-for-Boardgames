from datetime import datetime
from random import random
import numpy as np

from experiment import Experiment
from TicTacToe.players.reinforcePlayer import ReinforcePlayer, PGStrategy
from TicTacToe.players.base_players import ExperiencedPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.evaluation import evaluate_against_base_players
from plotting import Printer


def evaluate_against_each_other(player1, player2):
    """
    Evaluates player1 vs player2 using direct matches in oder to determine which one is used as new reference player
    :param player1:
    :param player2:
    :return: True if player1 scores at least as high as player2
    """
    score, results = evaluate_against_base_players(player1, [player2])
    return score >= 0


def evaluate_both_players(player1, player2):
    """
    Evaluates both player2 and player1 against base players and each other to determine which one is used as new reference player.

    :param player1:
    :param player2:
    :return: True if player1 scores at least as high as player2
    """
    score, results = evaluate_against_base_players(player1, [player2])
    p1_score, results = evaluate_against_base_players(player1)
    p2_score, results = evaluate_against_base_players(player2)
    p1_score += score
    p2_score -= score

    return p1_score >= p2_score


class TrainReinforcePlayerVsBest(Experiment):

    def __init__(self, games, evaluations, pretrained_player=None):
        super(TrainReinforcePlayerVsBest, self).__init__()
        self.games = games
        self.evaluations = evaluations
        self.pretrained_player = pretrained_player.copy(shared_weights=False) if pretrained_player else None

        self.__plotter__.line3_name = "ExperiencedPlayer score"

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations, pretrained_player=self.pretrained_player)
        return self

    def run(self, lr, batch_size, silent=False):
        self.player1 = self.pretrained_player if self.pretrained_player else ReinforcePlayer(PGStrategy, lr=lr, batch_size=batch_size)

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
            score, results = evaluate_against_base_players(self.player1)
            self.add_results(results)

            if not silent:
                if Printer.print_episode(episode*games_per_evaluation, self.games, datetime.now() - start_time):
                    self.plot_and_save(
                        "ReinforcementTraining LR: %s" % lr,
                        "Train ReinforcementPlayer vs self with shared network\nLR: %s Games: %s \nFinal score: %s" % (lr, episode*games_per_evaluation, score))

            # TODO: Play P1 vs P2 or evaluate both P1 and P2 against each other and all base players for evaluating strongest player?
            if evaluate_against_each_other(self.player1, self.player2):
            # if evaluate_both_players(self.player1, self.player2):
                self.player2 = self.player1.copy(shared_weights=False)
                self.replacements.append(episode)

        print("Best player replaced after episodes: %s" % self.replacements)
        self.final_score, self.final_results = evaluate_against_base_players(self.player1, silent=False)
        return self


if __name__ == '__main__':

    GAMES = 1000000
    EVALUATIONS = 1000
    LR = random()*1e-9 + 2e-5
    BATCH_SIZE = 32

    PLAYER = None  # Experiment.load_player("ReinforcePlayer using 3 layers pretrained on legal moves for 1000000 games.pth")

    experiment = TrainReinforcePlayerVsBest(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER)
    experiment.run(lr=LR, batch_size=BATCH_SIZE)

    print("\nSuccessfully trained on %s games" % experiment.num_episodes)
    if PLAYER:
        print("Pretrained on %s legal moves" % 1000000)