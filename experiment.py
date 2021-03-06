import os
import torch
from abc import ABC, abstractmethod

from plotting import Plotter


class Experiment(ABC):
    """ Base class for running experiments. Provides a plotter as well as path handling. DO NOT FORGET TO CALL super()"""

    def __init__(self):
        self.experiment_name = self.__class__.__name__
        self.__plotter__ = Plotter()
        self.last_plot = None

        self.path = self.config.TIC_TAC_TOE_DIR + "/experiments/artifacts/%s/" % self.experiment_name

    @abstractmethod
    def run(self, silent=False):
        pass

    @abstractmethod
    def reset(self):
        pass

    @classmethod
    def load_player(cls, player_name):
        filename = cls.config.find_in_subdirectory(player_name, cls.config.TIC_TAC_TOE_DIR + "/experiments")
        return torch.load(filename)

    def add_results(self, results):
        """
        Takes a single tuple or a list of tuples (name, value) and appends them to the internal plotter.
        Each distinct name is plotted as separately with its values interpolated to fit the other values.

        :param results: a list of tuples (name, value)
        :return: None
        """
        if not self.__plotter__:
            raise Exception("__plotter__ not initialized, Experiment's super() must be called")
        try:
            if isinstance(results, list):
                self.__plotter__.add_values(results)
            elif isinstance(results, tuple):
                self.__plotter__.add_values([results])
        except Exception as e:
            raise Exception("add_result received an illegal argument: " + str(e))

    def add_loss(self, loss):
        if not self.__plotter__:
            raise Exception("__plotter__ not initialized, Experiment's super() must be called")
        self.__plotter__.add_loss(loss)

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

    def save_player(self, player, description=""):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        torch.save(player, self.path + player.__str__() + " " + description + ".pth")

    @property
    def num_episodes(self):
        return self.__plotter__.num_episodes

    def __str__(self):
        return self.__class__.__name__

    class AlternatingColorIterator:
        """
        Returns Black and White alternately, starting with WHITE
        """
        def __init__(self):
            from Othello.config import BLACK, WHITE
            self.colors = [BLACK, WHITE]

        def __iter__(self):
            return self

        def __next__(self):
            self.colors = list(reversed(self.colors))
            return self.colors[0]
