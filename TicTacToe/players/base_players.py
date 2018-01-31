import numpy as np
from random import choice, random

import TicTacToe.config as config
from abstractClasses import Player
from TicTacToe.environment.game import TicTacToe


class RandomPlayer(Player):

    @staticmethod
    def get_move(board):
        valid_moves = board.get_valid_moves()
        return choice(valid_moves)

    def register_winner(self, winner_color):
        """ End of episode callback. @return the accumulated loss for the episode if available"""
        pass


class ExperiencedPlayer(Player):
    """ Wins games, blocks opponent, uses Heuristic Table """
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
    """ Never loses, only draws """
    pass

    def get_move(self, board):
        pass

    def register_winner(self, winner_color):
        """ End of episode callback. @return the accumulated loss for the episode if available"""
        pass


def evaluate_against_base_players(player):
    """ Standardized evaluation against base players. @return an evaluation score (0, 100) """

    EVALUATION_PLAYERS = (RandomPlayer(), ExperiencedPlayer())

    try:
        player.strategy.train = True
    except AttributeError:
        pass

    accumulated_results = []
    for e_player in EVALUATION_PLAYERS:
        simulation = TicTacToe([player, e_player])
        results, losses = simulation.run_simulations(config.EVALUATION_GAMES)
        accumulated_results.append(results)

    return np.mean(accumulated_results)
