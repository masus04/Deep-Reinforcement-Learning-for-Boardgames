import sys
import os
import math
import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd


class Plotter:

    def __init__(self):
        HISTORY_SIZE = 1000

        self.num_episodes = 0
        self.losses = DataResolutionManager([], storage_size=HISTORY_SIZE)
        self.scores = DataResolutionManager([], storage_size=HISTORY_SIZE)

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_score(self, score):
        self.num_episodes += 1
        self.scores.append(score)

    def plot(self, title):

        line1_name = "losses"
        line1_values = self.losses.get_values()

        line2_name = "score"
        line2_values = self.scores.get_values()

        if not line1_values or not line2_values:
            raise Exception("Cannot plot empty values")

        line1 = pd.Series(line1_values, name=line1_name)
        line2 = pd.Series(line2_values, name=line2_name)
        df = pd.DataFrame([line1, line2])
        df = df.transpose()
        df.plot(secondary_y=[line2_name], legend=True, figsize=(16, 9))

        plt.title(title)
        plt.xlabel = "Episodes"

        return plt
        plt.savefig("./plots" + path + "%s.png" % file_name)


class DataResolutionManager:

    def __init__(self, data_points=[], storage_size=1000):
        try:  # data_points can be either DataResolutionManagers or simple lists
            data_points = data_points.get_values()
        except AttributeError:
            pass

        data_points = data_points.copy()
        self.storage_size = storage_size
        self.compression_factor = math.ceil((len(data_points)+1) / storage_size)
        self.values = []
        self.buffer = []

        if self.compression_factor > 1:
            self.buffer = data_points[len(data_points) - len(data_points) % self.compression_factor:]

            for i in range(len(data_points) // self.compression_factor):
                self.values.append(sum(data_points[:self.compression_factor]) / self.compression_factor)
                data_points = data_points[self.compression_factor:]
        else:
            self.values = data_points

    def append(self, value):
        self.buffer.append(value)
        if len(self.buffer) >= self.compression_factor:
            self.values.append(sum(self.buffer) / len(self.buffer))
            self.buffer = []
            if len(self.values) >= 2*self.storage_size:
                if len(self.values) % 2 != 0:  # Move uneven element back to buffer
                    self.buffer.append(self.values.pop())
                self.values = [(a + b) / 2 for a, b in zip(self.values[0::2], self.values[1::2])]
                self.compression_factor *= 2

    def get_values(self):
        if len(self.buffer) == 0:
            return self.values
        else:
            return self.values + [sum(self.buffer) / len(self.buffer)]


class Printer:

    @staticmethod
    def print_inplace(text, percentage, time_taken=None, comment=""):
        percentage = int(percentage)
        length_factor = 5
        progress_bar = int(round(percentage/length_factor)) * "*" + (round((100-percentage)/length_factor)) * "."
        progress_bar = progress_bar[:round(len(progress_bar)/2)] + "|" + str(int(percentage)) + "%|" + progress_bar[round(len(progress_bar)/2):]
        sys.stdout.write("\r%s |%s|" % (text, progress_bar) + (" Time: %s" % str(time_taken).split(".")[0] if time_taken else "") + comment)
        sys.stdout.flush()

        if percentage == 100:
            print()