import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod

import TicTacToe.config as config


class Board(ABC):
    """
    Represents an interface for the board of any game board used in this package.
    """
    @abstractmethod
    def get_valid_moves(self, color):
        """ Returns a list of valid moves for the player represented with :param color.

        :param color: The color used to represent the player for which to search valid moves
        :return: A list valid moves. The representation of moves is determined by the implemented game.
        """
        pass

    @abstractmethod
    def apply_move(self, move, color):
        """
        Applies a move to the board including all effects to the rest of the board the move may have.

        The move can be illegal, depending on the game implementation. But if so, this method should handle the consequences.

        :param move: The move to be applied
        :param color: The color used to represent the player which performs the move
        :return: self
        """
        pass

    @abstractmethod
    def game_won(self):
        """
        Checks if the game has ended. If so, return the winner.

        :return: The winner. None if the game is not yet over.
        """
        pass

    @abstractmethod
    def get_representation(self, color):
        """
        Generates a representation of the board in which black is always the current player.

        :param color: The color used to represent the player in game
        :return: A copy of the (modified) game state
        """
        pass

    @abstractmethod
    def get_legal_moves_map(self, color):
        """
        Generates a map of the game state with 0 everywhere except for the tiles where moves are valid.

        This is useful for matrix operations on the board representation

        :param color: The color used to represent the player
        :return: A board state where only legal moves are nonzero
        """
        pass

    @abstractmethod
    def copy(self):
        """
        Returns a copy of the board and its state so that further manipulations do not interfere with the original object.

        :return: a copy of the board and its state.
        """
        pass

    @staticmethod
    def other_color(color):
        """
        Determines the opponents color depending on the given color. Empty if :param color == Empty.

        :param color: The color used to represent the player
        :return: The color of the opposing player
        """
        if color == config.BLACK:
            return config.WHITE
        elif color == config.WHITE:
            return config.BLACK
        elif color == config.EMPTY:
            return config.EMPTY
        else:
            raise BoardException("Illegal color provided: %s" % color)


class BoardException(Exception):
    """
    BoardException is raised if Board encounters an illegal state.
    """
    pass


class Player(ABC):
    """
    Abstract base class for all players of any game used in this package. Each player implements its own strategy.

    Players are hooked into their specific game frameworks. The game defines the format of the Board and move.
    """
    def __init__(self):
        super(Player, self).__init__()
        self.color = None
        self.original_color = None
        self.num_moves = 0

    @abstractmethod
    def get_move(self, board):
        """
        Core method of the player: given a board provide the next move to apply to it following a strategy given by the player.

        :param board: The board for which to decide on a move to take.
        :return: a move in a format defined by the game implementation.
        """
        pass

    def register_winner(self, winner_color):
        """
        Callback method which is called after a game has ended. Announces the winner of the game.

        :param winner_color: The color used to represent the winner of the game that just finished.
        :return: A loss measure for the whole game if available. This is used mostly for plotting statistics.
        """
        return 0

    def save(self):
        """
        Save the current player including the state of training if applicable.

        :return: None
        """
        pass

    def get_label(self, winner_color):
        """
        Produce a label given the players current state and the winner's color.

        :param winner_color: The color used to represent the winner
        :return: A label that can be used in training.
        """
        if self.original_color == winner_color:
            return config.LABEL_WIN
        elif Board.other_color(self.original_color) == winner_color:
            return config.LABEL_LOSS
        else:
            return config.LABEL_DRAW

    def __str__(self):
        return "[%s]" % self.__class__.__name__


class LearningPlayer(Player):

    def __init__(self, strategy):
        super(LearningPlayer, self).__init__()

        if issubclass(strategy.__class__, Strategy):
            self.strategy = strategy
        else:
            raise Exception("ReinforcePlayer takes as a strategy argument a subclass of %s, received %s" % (Model, strategy))

    def get_move(self, board):
        if self.strategy.train:
            self.strategy.rewards.append(0)
        return self.strategy.evaluate(board.get_representation(self.color), board.get_legal_moves_map(self.color))

    def register_winner(self, winner_color):
        if self.strategy.train:
            self.strategy.rewards[-1] = self.get_label(winner_color)

        return self.strategy.update()

    def copy(self, shared_weights=True):
        """
        Returns a clean copy of the player and all its attributes.

        :param shared_weights: If True, the returned player shares a Model and therefore the weights with the original player
        :return:
        """
        return self.__class__(strategy=self.strategy.copy(shared_weights=shared_weights))

    def __str__(self):
        return "[%s lr:%s]" % (self.__class__.__name__, self.strategy.lr)


class PlayerException(Exception):
    pass


class Strategy(ABC):
    """
    Abstract Base class for all complex strategies used by any player.

    The strategy contains and trains a model that is capable of
    """

    def __init__(self):
        self.lr = None
        self.model = None
        self.optimizer = None
        self.rewards = []
        self.log_probs = []
        self.train = True

    @abstractmethod
    def evaluate(self, board_sample):
        pass

    @abstractmethod
    def update(self):
        pass

    def copy(self, shared_weights=True):
        if shared_weights:
            strategy = self.__class__(model=self.model, lr=self.lr, batch_size=self.batch_size)
        else:
            strategy = self.__class__(model=self.model.copy(), lr=self.lr, batch_size=self.batch_size)

        strategy.train = deepcopy(self.train)
        strategy.log_probs = deepcopy(self.log_probs)
        return strategy

    @staticmethod
    def discount_rewards(rewards, discount_factor):
        if discount_factor <= 0:
            return deepcopy(rewards)

        running_reward = 0  # rewards[-1]
        discounted_rewards = []
        for r in rewards[::-1]:
            running_reward = (discount_factor * running_reward + r)
            discounted_rewards.insert(0, running_reward)

        return discounted_rewards

    @staticmethod
    def normalize_rewards(rewards):
        return (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps)

    def bootstrap_rewards(self):
        # TODO: Catch illegal use of this method
        pred_values = [self.model(config.make_variable([self.board_samples[i]]), config.make_variable([self.legal_moves[i]]))[1].data[0,0] for i in range(len(self.board_samples))]
        pred_values[-1] = self.rewards[-1]

        rewards = [config.ALPHA * pred_values[i+1] - pred_values[i] for i in range(len(pred_values)-1)]
        rewards.append(self.rewards[-1])

        return rewards

    def rewards_baseline(self, rewards):
        # TODO: Catch illegal use of this method
        rewards = [rewards[i] - self.model(config.make_variable([self.board_samples[i]]), config.make_variable([self.legal_moves[i]]))[1].data[0, 0] for i in range(len(self.board_samples))]
        return rewards


class Model(torch.nn.Module):

    def __xavier_initialization__(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal(module.weight.data)
                # torch.nn.init.xavier_normal(module.bias.data)

    @staticmethod
    def legal_softmax(input, legal_moves_map):
        legal_moves_map = legal_moves_map.view(input.data.shape)
        x = input

        # set illegal moves to about float.minvalue
        x = x * legal_moves_map
        x = x + (legal_moves_map-1) * 3e38

        x = F.softmax(x, dim=1)
        return x

    def copy(self):
        return deepcopy(self)
