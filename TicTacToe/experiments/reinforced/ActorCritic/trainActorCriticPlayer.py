from datetime import datetime
from random import random
import numpy as np

from experiment import Experiment
from TicTacToe.players.actorCriticPlayer import FCActorCriticPlayer
from TicTacToe.players.base_players import ExperiencedPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.evaluation import evaluate_against_base_players
from plotting import Printer


class TrainActorCriticPlayer(Experiment):

    def __init__(self, games, evaluations, pretrained_player=None):
        super(TrainActorCriticPlayer, self).__init__()
        self.games = games
        self.evaluations = evaluations
        self.pretrained_player = pretrained_player.copy(shared_weights=False) if pretrained_player else None

        self.__plotter__.line3_name = "ExperiencedPlayer score"

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations, pretrained_player=self.pretrained_player)
        return self

    def run(self, lr, batch_size, silent=False):

        self.player1 = self.pretrained_player if self.pretrained_player else FCActorCriticPlayer(lr=lr, batch_size=batch_size)

        # Player2 shares the same weights but does not change them.
        self.player2 = self.player1.copy(shared_weights=True)
        self.player2.strategy.train = False

        self.simulation = TicTacToe([self.player1, self.player2])

        games_per_evaluation = self.games // self.evaluations
        start_time = datetime.now()
        for episode in range(1, self.evaluations+1):
            # train
            self.player1.strategy.train, self.player1.strategy.policy.training, self.player1.strategy.value_function.training = True, True, True  # training mode

            results, losses = self.simulation.run_simulations(games_per_evaluation)
            self.add_results(("Losses", np.mean(losses)))

            # evaluate
            self.player1.strategy.train, self.player1.strategy.policy.training, self.player1.strategy.value_function.training = False, False, False  # eval mode
            score, results = evaluate_against_base_players(self.player1)
            self.add_results(results)

            if not silent:
                if Printer.print_episode(episode*games_per_evaluation, self.games, datetime.now() - start_time):
                    self.plot_and_save(
                        "ReinforcementTraining LR: %s" % lr,
                        "Train %s vs self with shared network\nLR: %s Games: %s \nFinal score: %s" % (self.player1, lr, episode*games_per_evaluation, results))

        self.final_score, self.final_results = evaluate_against_base_players(self.player1, silent=False)
        return self


if __name__ == '__main__':

    GAMES = 10000
    EVALUATIONS = 100
    LR = random()*1e-9 + 1e-4
    BATCH_SIZE = 32

    PLAYER = None  # Experiment.load_player("Pretrain player [all traditional opponents].pth")

    print("Training ReinforcePlayer vs self with lr: %s" % LR)
    experiment = TrainActorCriticPlayer(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER)
    experiment.run(lr=LR, batch_size=BATCH_SIZE)

    print("\nSuccessfully trained on %s games" % experiment.num_episodes)
    if PLAYER:
        print("Pretrained on %s legal moves" % 1000000)
