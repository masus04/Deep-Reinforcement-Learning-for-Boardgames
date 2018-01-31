import os
import torch
from abc import ABC, abstractmethod

import TicTacToe.config as config
from plotting import Plotter


class Experiment(ABC):
    """ Base class for running experiments. Provides a plotter as well as path handling. DO NOT FORGET TO CALL super()"""

    def __init__(self, experiment_path):
        self.experiment_name = self.__class__.__name__
        self.__plotter__ = Plotter()
        self.last_plot = None

        self.path = "%s/%s/" % (experiment_path, self.experiment_name)
        print(self.path)

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def load_player(player_name):
        raise NotImplementedError("Implement this when needed")

    def add_losses(self, losses):
        if not self.__plotter__:
            raise Exception("__plotter__ not initialized, Experiment's super() must be called")
        for l in losses:
            self.__plotter__.add_loss(l)

    def add_scores(self, score, second_score=None):
        if not self.__plotter__:
            raise Exception("__plotter__ not initialized, Experiment's super() must be called")
        self.__plotter__.add_score(score, second_score)

    def plot_scores(self, title):
        if not self.__plotter__:
            raise Exception("__plotter__ not initialized, Experiment's super() must be called")
        self.last_plot = self.__plotter__.plot(title)

    def plot_and_save(self, file_name, plot_title=""):
        self.plot_scores(plot_title if plot_title else file_name)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.last_plot.savefig(self.path + file_name + ".png")
        self.last_plot.close("all")

    def save_player(self, player, filename):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        torch.save(player, self.path + filename)

    class AlternatingColorIterator:
        """
        Returns Black and White alternatingly, starting with WHITE
        """
        def __init__(self):
            self.colors = [config.BLACK, config.WHITE]

        def __iter__(self):
            return self

        def __next__(self):
            self.colors = list(reversed(self.colors))
            return self.colors[0]
