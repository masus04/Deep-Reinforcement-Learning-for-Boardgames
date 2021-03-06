import sys
import math
import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict


class Plotter:

    def __init__(self):
        self.values = OrderedDict()
        self.losses = DataResolutionManager()

    def add_values(self, values):
        """
        Accepts a list of tuples (score_name, score) and adds it to the internal representation to be plotted

        :param scores: a list of tuples (score_name, score)
        :return: None
        """
        for val in values:
            try:
                self.values[val[0]].append(val[1])
            except KeyError:
                self.values[val[0]] = DataResolutionManager()
                self.values[val[0]].append(val[1])

    def add_loss(self, losses):
        self.losses.append(losses)

    def plot(self, title):

        fig, axes = plt.subplots(nrows=2, ncols=1)

        if self.values:
            lines = []
            max_length = max([len(values[1].get_values()) for values in self.values.items()])
            for key in self.values:
                values = self.__scale__(self.values[key].get_values(), max_length+1)
                series = pd.Series(values, name=key)
                lines.append(series)
            df = pd.DataFrame(lines)

        # DEPRECATED
        else:
            line1_values = self.losses.get_values()

            line2_values = self.scores.get_values()
            line2_values = self.__scale__(line2_values, len(line1_values))

            line3_values = self.second_scores.get_values()
            line3_values = self.__scale__(line3_values, len(line1_values))

            if len(line1_values) == 0 and len(line2_values) == 0:
                raise Exception("Cannot plot empty values losses and scores")

            line1 = pd.Series(line1_values, name=self.line1_name)
            line2 = pd.Series(line2_values, name=self.line2_name)
            line3 = pd.Series(line3_values, name=self.line3_name)
            df = pd.DataFrame([line1, line2, line3])

        if len(self.losses) > 0:
            line = pd.Series(self.losses.get_values(), name="Loss")
            loss_frame = pd.DataFrame(line)
            loss_ax = loss_frame.plot(ax=axes[0], title=title, legend=True, figsize=(16, 9))
            loss_ax.xaxis.set_ticks([i * self.num_episodes / 10 for i in range(11)])
            loss_ax.set_xticklabels([i * self.num_episodes / 10 for i in range(11)])
        else:
            plt.title(title)

        df = df.transpose()
        ax = df.plot(ax=axes[1], legend=True, figsize=(16, 9), ylim=(-1, 1) if len(self.losses) > 0 else None)  # secondary_y=[line2_name] for separate scales | ylim=(min, max) for limiting y scale

        ax.xaxis.set_ticks([i * self.num_episodes / 10 for i in range(11)])
        ax.set_xticklabels([i * self.num_episodes / 10 for i in range(11)])
        ax.yaxis.set_ticks = [i/2 for i in range(-2, 3)]
        ax.axhline(y=0, color='black', linewidth=2)
        ax.axhline(y=0.5, color='black', linewidth=1)
        ax.axhline(y=-0.5, color='black', linewidth=1)

        plt.xlabel("Episodes")
        plt.grid()

        return plt

    @staticmethod
    def __scale__(lst, length):
        if len(lst) > 1:
            old_indices = np.arange(0, len(lst))
            new_indices = np.linspace(0, len(lst) - 1, length)
            spl = UnivariateSpline(old_indices, lst, k=1, s=0)
            lst = spl(new_indices)

        return lst

    @property
    def num_episodes(self):
        return max((len(self.values[value]) for value in self.values))


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

    def get_real_length(self):
        return len(self.values) * self.compression_factor + len(self.buffer)

    def __str__(self):
        return self.get_values().__str__()

    def __len__(self):
        return len(self.values)


class Printer:

    @staticmethod
    def print_episode(episode, total_episodes, time_taken=None, print_every_iteration=False):
        """
        Prints progress every full percent of the episode.

        :param episode: The current episode number
        :param total_episodes: The total episode number
        :param time_taken: A datetime diff eg. start_time - datetime.now()
        :param print_every_iteration: Flag that lets progress be printed every episode instead of only on full percentages
        :return: True if progress was printed, False otherwise
        """
        """ Prints progress on the current episode.
            Only prints full percentages unless specified otherwise using the @print_every_iteration flag"""
        if print_every_iteration or 100 * episode/total_episodes % 1 == 0:
            Printer.print_inplace("Episode %s/%s" % (episode, total_episodes), 100 * episode // total_episodes, time_taken)
            return True

        return False

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
