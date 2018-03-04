from datetime import datetime
from random import random, choice
import numpy as np

from experiment import Experiment
from TicTacToe.players.base_players import RandomPlayer, NovicePlayer, ExperiencedPlayer
from TicTacToe.players.reinforcePlayer import FCReinforcePlayer
from TicTacToe.players.actorCriticPlayer import FCActorCriticPlayer
from TicTacToe.players.proximalPolicyOptimizationPlayer import FCPPOPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.evaluation import evaluate_against_base_players
from plotting import Printer


class TrainReinforcePlayerVsTraditionalOpponent(Experiment):

    def __init__(self, games, evaluations, pretrained_player, opponent):
        super(TrainReinforcePlayerVsTraditionalOpponent, self).__init__()
        self.games = games
        self.evaluations = evaluations
        self.pretrained_player = pretrained_player.copy(shared_weights=False) if pretrained_player else None
        self.opponent = opponent

        self.__plotter__.line3_name = "opponent score"

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations, pretrained_player=self.pretrained_player, opponent=self.opponent)
        return self

    def run(self, lr, batch_size, silent=False):

        self.player1 = self.pretrained_player if self.pretrained_player else FCPPOPlayer(lr=lr, batch_size=batch_size)
        if self.opponent is not None:
            self.player2 = self.opponent
            self.simulation = TicTacToe([self.player1, self.player2])

        games_per_evaluation = self.games // self.evaluations
        start_time = datetime.now()
        for episode in range(1, self.evaluations+1):

            if self.opponent is None:
                self.player2 = choice((RandomPlayer(), NovicePlayer(), ExperiencedPlayer(deterministic=False, block_mid=True)))
                self.simulation = TicTacToe([self.player1, self.player2])

            # train
            try:
                self.player1.strategy.train, self.player1.strategy.model.training = True, True  # training mode
            except AttributeError:
                self.player1.strategy.train, self.player1.strategy.policy.training, self.player1.strategy.value_function.training = True, True, True  # training mode

            results, losses = self.simulation.run_simulations(games_per_evaluation)
            self.add_results(("Losses", np.mean(losses)))
            # self.add_loss(sum(losses) / len(losses))    # losses are interesting during training

            # evaluate
            try:
                self.player1.strategy.train, self.player1.strategy.model.training = False, False  # training mode
            except AttributeError:
                self.player1.strategy.train, self.player1.strategy.policy.training, self.player1.strategy.value_function.training = False, False, False  # training mode


            score, results = evaluate_against_base_players(self.player1)
            self.add_results(results)
            # self.add_scores(main_score, opponent_score)

            if not silent:
                if Printer.print_episode(episode*games_per_evaluation, self.games, datetime.now() - start_time):
                    self.plot_and_save(
                        "%s vs %s LR: %s" % (self.player1, self.opponent, lr),
                        "Train %s vs traditional opponents: %s \nLR: %s Games: %s \nFinal score: %s" % (self.player1, self.opponent, lr, episode*games_per_evaluation, results))

        self.final_score, self.final_results = evaluate_against_base_players(self.player1, silent=False)
        return self


if __name__ == '__main__':

    GAMES = 100000
    EVALUATIONS = 1000
    LR = random()*1e-9 + 1e-4
    BATCH_SIZE = 32

    PLAYER = None  # Experiment.load_player("ReinforcePlayer using 3 layers pretrained on legal moves for 1000000 games.pth")
    OPPONENT = None  # ExperiencedPlayer(deterministic=False, block_mid=False)

    print("Training ReinforcePlayer vs %s with lr: %s" % (OPPONENT, LR))
    experiment = TrainReinforcePlayerVsTraditionalOpponent(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER, opponent=OPPONENT)
    experiment.run(lr=LR, batch_size=BATCH_SIZE)
    experiment.save_player(experiment.player1, "%s pretrained on traditional opponents" % experiment.player1)
    print("Successfully trained on %s games, pretrained on %s" % (experiment.__plotter__.num_episodes, 10000000))

