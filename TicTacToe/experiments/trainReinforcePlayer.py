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

    def run(self, lr, silent=False):

        self.player1 = ReinforcePlayer(PGStrategy, lr=lr)

        # Player2 shares the same weights but does not change them.
        self.player2 = self.player1.copy(shared_weights=True)
        self.player2.strategy.train = False

        self.simulation = TicTacToe([self.player1, self.player2])


        start_time = datetime.now()

        for episode in range(self.evaluations):
            # train
            self.player1.strategy.train = True
            results, losses = self.simulation.run_simulations(self.games // self.evaluations)
            self.add_loss(sum(losses) / len(losses))    # losses are interesting during training

            # evaluate
            self.player1.strategy.train = False
            result = evaluate_against_base_players(self.player1)
            self.add_scores(result)    # only evaluation results are interesting

            if not silent:
                if Printer.print_episode(episode+1, self.evaluations, datetime.now() - start_time):
                    self.plot_and_save("ReinforcementTraining LR: %s" % lr, "ReinforcementTraining \nLR: %s Games: %s \nFinal score: %s" % (lr, episode, result))

        return self


if __name__ == '__main__':

    GAMES = 100000
    EVALUATIONS = 100
    LR = 1e-4

    experiment = TrainReinforcePlayer(games=GAMES, evaluations=EVALUATIONS)
    experiment.run(lr=LR)

    print("Successively trained on %s games" % experiment.__plotter__.num_episodes)

