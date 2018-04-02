import numpy as np
from random import choice, random

import Othello.config as config
from abstractClasses import Player, PlayerException
from Othello.players.heuristics import OthelloHeuristic
from Othello.players.search_based_ai import GameArtificialIntelligence


class RandomPlayer(Player):
    """
    Applies a random valid move
    """
    def get_move(self, board):
        try:
            return choice(list(board.get_valid_moves(self.color)))
        except IndexError:
            return None


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

        try:
            return choice(list(board.get_valid_moves(self.color)))
        except IndexError:
            return None


class ExperiencedPlayer(Player):
    """ Wins games or blocks opponent with the next move. Uses Heuristic Table if there are no winning or blocking moves"""
    heuristic_table = np.array([[100, -25, 10,  5,  5, 10, -25, 100],
                                [-25, -25,  2,  2,  2,  2, -25, -25],
                                [ 10,   2,  5,  1,  1,  5,   2,  10],
                                [  5,   2,  1,  2,  2,  1,   2,   5],
                                [  5,   2,  1,  2,  2,  1,   2,   5],
                                [ 10,   2,  5,  1,  1,  5,   2,  10],
                                [-25, -25,  2,  2,  2,  2, -25, -25],
                                [100, -25, 10,  5,  5, 10, -25, 100]])

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

        try:
            if denies:
                return max(denies)[1]
            else:
                return max(attacks)[1]
        except ValueError:
            return None

    def evaluate_heuristic_table(self, board):
        self_mask = board.board == self.color
        other_mask = board.board == board.other_color(self.color)
        score = np.sum(self.heuristic_table * self_mask - self.heuristic_table * other_mask)
        if not self.deterministic:
            score += random() * 0.001  # Introduce some randomness to equally valued boards
        return score


class ExpertPlayer(Player):
    """ Perfect player: never loses, only draws """
    pass

    def get_move(self, board):
        raise NotImplementedError("Implement when needed")


class SearchPlayer(Player):

    def __init__(self, time_limit=5, strategy=OthelloHeuristic.DEFAULT_STRATEGY):
        super(SearchPlayer, self).__init__()
        self.time_limit = time_limit
        self.ai = GameArtificialIntelligence(OthelloHeuristic(strategy).evaluate)

    def get_move(self, board):
        return self.ai.move_search(board, self.time_limit, self.color, board.other_color(self.color))
