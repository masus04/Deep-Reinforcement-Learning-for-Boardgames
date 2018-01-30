import torch
from abc import ABC, abstractmethod

import TicTacToe.config as config

class Board(ABC):

    @abstractmethod
    def get_valid_moves(self, color):
        pass

    @abstractmethod
    def apply_move(self, move, color):
        pass

    @abstractmethod
    def game_won(self):
        pass

    @abstractmethod
    def get_representation(self, color):
        pass

    @abstractmethod
    def get_legal_moves_map(self, color):
        pass

    @abstractmethod
    def copy(self):
        pass

    @staticmethod
    def other_color(color):
        if color == config.BLACK:
            return config.WHITE
        if color == config.WHITE:
            return config.BLACK
        if color == config.EMPTY:
            return color
        raise BoardException("Illegal color provided: %s" % color)


class BoardException(Exception):
    pass


class Player(ABC):

    color = None
    original_color = None

    @abstractmethod
    def get_move(self, board):
        pass

    def register_winner(self, winner_color):
        pass

    def save(self):
        pass

    def get_label(self, winner_color):
        if self.original_color == winner_color:
            return config.LABEL_WIN
        if Board.other_color(self.color) == winner_color:
            return config.LABEL_LOSS
        return config.LABEL_DRAW


class Strategy(ABC):

    def __init__(self):
        self.lr = None
        self.model = None
        self.optimizer = None
        self.training_samples = []
        self.train = True

    @abstractmethod
    def evaluate(self, board_sample):
        pass

    @abstractmethod
    def update(self, training_labels):
        pass


class Model(torch.nn.Module):
    pass
