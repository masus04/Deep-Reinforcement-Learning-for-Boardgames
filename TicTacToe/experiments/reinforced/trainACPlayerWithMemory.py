from datetime import datetime
from random import random, randint, choice, uniform
import numpy as np

from experiment import Experiment
from TicTacToe.players.acPlayer import FCACPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.environment.evaluation import evaluate_against_base_players
from plotting import Printer


class TrainACPlayerWithMemory(Experiment):

    def __init__(self, games, evaluations, pretrained_player=None):
        super(TrainACPlayerWithMemory, self).__init__()
        self.games = games
        self.evaluations = evaluations
        self.pretrained_player = pretrained_player.copy(shared_weights=False) if pretrained_player else None

        self.__plotter__.line3_name = "ExperiencedPlayer score"

    def reset(self):
        self.__init__(games=self.games, evaluations=self.evaluations, pretrained_player=self.pretrained_player)
        return self

    def run(self, lr, batch_size, silent=False, p=0.01):

        self.player1 = self.pretrained_player if self.pretrained_player else FCACPlayer(lr=lr, batch_size=batch_size)
        player_memory = []
        memory_history = []

        # Player2 shares the same weights but does not change them.
        self.player2 = self.player1.copy(shared_weights=True)
        self.player2.strategy.train = False

        self.simulation = TicTacToe([self.player1, self.player2])

        games_per_evaluation = self.games // self.evaluations
        start_time = datetime.now()
        for episode in range(1, self.evaluations+1):
            # train
            self.player1.strategy.train, self.player1.strategy.model.training = True, True  # training mode

            if random() > p or len(player_memory) == 0:
                results, losses = self.simulation.run_simulations(games_per_evaluation)
            else:
                memory_simulation = TicTacToe([self.player1, choice(player_memory)])
                results, losses = memory_simulation.run_simulations(games_per_evaluation)
            self.add_results(("Losses", np.mean(losses)))

            # evaluate
            self.player1.strategy.train, self.player1.strategy.model.training = False, False  # eval mode
            score, results = evaluate_against_base_players(self.player1)
            self.add_results(results)

            if not silent:
                if Printer.print_episode(episode*games_per_evaluation, self.games, datetime.now() - start_time):
                    self.plot_and_save(
                        "ReinforcementTraining LR: %s" % lr,
                        "Train ACPlayer vs self with shared network\nLR: %s Games: %s p: %s" % (lr, episode*games_per_evaluation, p))

            if random() < p:  # With a certain probability, keep weights as a reference for later training
                if len(player_memory) > 20:
                    player_memory.pop(randint(0, 9))
                memory_player = self.player1.copy(shared_weights=False)
                memory_player.strategy.train, memory_player.strategy.model.training = False, False
                memory_player.mem_episode = episode
                player_memory.append(memory_player)
                memory_history.append([memory.mem_episode for memory in player_memory])

        print(memory_history)
        self.final_score, self.final_results = evaluate_against_base_players(self.player1, silent=False)
        return self


if __name__ == '__main__':

    ITERATIONS = 10
    start = datetime.now()

    for i in range(ITERATIONS):
        print("Iteration %s/%s" % (i+1, ITERATIONS))
        GAMES = 500000
        EVALUATIONS = 1000
        LR = uniform(1e-3, 1e-5)  # random()*1e-9 + 1e-4
        BATCH_SIZE = 1
        P = uniform(1e-1, 1e-3)

        PLAYER = None  # Experiment.load_player("Pretrain player [all traditional opponents].pth")

        print("Training ACPlayer vs self with lr: %s" % LR)
        experiment = TrainACPlayerWithMemory(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER)
        experiment.run(lr=LR, batch_size=BATCH_SIZE, p=P)
        print()

    print("\nSuccessfully trained on %s games" % experiment.num_episodes)
    if PLAYER:
        print("Pretrained on %s legal moves" % 1000000)

    print("took %s" % datetime.now()-start)
