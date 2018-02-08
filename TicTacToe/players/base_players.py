import numpy as np
from random import choice, random

import TicTacToe.config as config
from abstractClasses import Player
from TicTacToe.environment.game import TicTacToe


class RandomPlayer(Player):
    """
    Applies a random valid move
    """
    @staticmethod
    def get_move(board):
        return choice(board.get_valid_moves())


class DeterministicPlayer(Player):
    """
    Very simple deterministic player that always selects the first possible move.

    This player is supposed to be as simple to beat as possible and should be used as a dummy opponent in training.
    """
    def get_move(self, board):
        return board.get_valid_moves(self.color)[0]


class NovicePlayer(Player):
    """
    Wins a game if possible in the next move, else applies a random move
    """
    def get_move(self, board):

        for move in board.get_valid_moves(self.color):
            afterstate = board.copy().apply_move(move, self.color)
            if afterstate.game_won() == self.color:
                return move

        valid_moves = board.get_valid_moves()
        return choice(valid_moves)


class ExperiencedPlayer(Player):
    """ Wins games or blocks opponent with the next move. Uses Heuristic Table if there are no winning or blocking moves"""
    heuristic_table = np.array([[1, 0.5, 1], [0.5, 0.75, 0.5], [1, 0.5, 1]])

    def __init__(self, deterministic=True, block_mid=False):
        self.deterministic = deterministic
        self.block_mid = block_mid

    def get_move(self, board):
        valid_moves = board.get_valid_moves(self.color)

        if self.block_mid and sum(board.count_stones()) == 1 and (1, 1) in valid_moves:
            return 1, 1

        denies, attacks = [], []
        for move in valid_moves:
            afterstate = board.copy().apply_move(move, self.color)
            if afterstate.game_won() == self.color:
                return move

            afterstate_opponent = board.copy().apply_move(move, board.other_color(self.color))
            if afterstate_opponent.game_won() == board.other_color(self.color):
                denies.append((self.evaluate_heuristic_table(afterstate_opponent), move))

            attacks.append((self.evaluate_heuristic_table(afterstate), move))

        if denies:
            return max(denies)[1]
        else:
            return max(attacks)[1]

    def evaluate_heuristic_table(self, board):
        self_mask = board.board == self.color
        other_mask = board.board == board.other_color(self.color)
        score = np.sum(self.heuristic_table * self_mask - self.heuristic_table * other_mask)
        if not self.deterministic:
            score += random() * 0.001  # Bring some randomness to equaly valued boards
        return score


class ExpertPlayer(Player):
    """ Perfect player: never loses, only draws """
    pass

    def get_move(self, board):
        raise NotImplementedError("Implement when needed")
