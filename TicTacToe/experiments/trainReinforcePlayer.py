import os
from datetime import datetime

from experiment import Experiment
from TicTacToe.players.reinforcePlayer import ReinforcePlayer, PGStrategy
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.evaluation import evaluate_against_base_players
from plotting import Printer


class TrainReinforcePlayer(Experiment):

    def __init__(self, games, evaluations):
        super(TrainReinforcePlayer, self).__init__(os.path.dirname(os.path.abspath(__file__)))
        self.games = games
        self.evaluations = evaluations

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations)
        return self

    def run(self, lr, batch_size, silent=False):

        self.player1 = ReinforcePlayer(PGStrategy, lr=lr, batch_size=batch_size)

        # Player2 shares the same weights but does not change them.
        self.player2 = self.player1.copy(shared_weights=True)
        self.player2.strategy.train = False

        self.simulation = TicTacToe([self.player1, self.player2])


        start_time = datetime.now()

        for episode in range(1, self.evaluations+1):
            # train
            self.player1.strategy.train, self.player1.strategy.model.training = True, True  # training mode
            games_per_evaluation = self.games // self.evaluations
            results, losses = self.simulation.run_simulations(games_per_evaluation)
            self.add_loss(sum(losses) / len(losses))    # losses are interesting during training

            # evaluate
            self.player1.strategy.train, self.player1.strategy.model.training = False, False  # eval mode
            result = evaluate_against_base_players(self.player1)
            self.add_scores(result)    # only evaluation results are interesting

            if not silent:
                if Printer.print_episode(episode*games_per_evaluation, self.games, datetime.now() - start_time):
                    self.plot_and_save(
                        "ReinforcementTraining LR: %s" % lr,
                        "Train ReinforcementPlayer vs self with shared network\nLR: %s Games: %s \nFinal score: %s" % (lr, episode, result))

        return self


if __name__ == '__main__':

    GAMES = 10000000
    EVALUATIONS = 1000
    LR = 10**-2
    BATCH_SIZE = 32

    experiment = TrainReinforcePlayer(games=GAMES, evaluations=EVALUATIONS)
    experiment.run(lr=LR, batch_size=BATCH_SIZE)

    print("Successfully trained on %s games" % experiment.__plotter__.num_episodes)

