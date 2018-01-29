
import os
from abc import ABC, abstractmethod

from plotting import Plotter


class Experiment(ABC):

    def __init__(self, game, players, experiment_name):
        self.experiment_name = experiment_name
        self.game = game
        self.players = players
        self.__plotter__ = Plotter()
        self.last_plot = None

        path = "./%s/" % experiment_name
        if not os.path.exists(path):
            os.makedirs(path)

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def load_player(player_name):
        pass

    def plot_scores(self, title):
        self.__plotter__.plot(title)

    def save_last_plot(self, file_name):
        if not self.last_plot:
            raise Exception("")
        self.last_plot.savefig(self.path + file_name)
