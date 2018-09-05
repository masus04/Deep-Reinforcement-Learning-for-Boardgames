import numpy as np
from random import choice, random
from numba import njit

import TicTacToe.config as config
from abstractClasses import Player, PlayerException


class PerfectPlayer(Player):

    def get_move(self, board):
        other_color = board.other_color(self.color)
        CORNERS = [(0, 0), (0, 2), (2, 0), (2, 2)]
        SIDES = [(1, 0), (0, 1), (1, 2), (2, 1)]

        # Win
        valid_moves = board.get_valid_moves(self.color)
        afterstates = [(move, board.copy().apply_move(move, self.color)) for move in valid_moves]  # chosen move, afterstate
        for move, afterstate in afterstates:
            if afterstate.game_won() == self.color:
                return move

        # Block win
        afterstates_opponent = []
        for move, afterstate in afterstates:
            opponent_vm = afterstate.get_valid_moves(other_color)
            opponent_as = [(move, afterstate.copy().apply_move(move, other_color)) for move in opponent_vm]
            afterstates_opponent += opponent_as

            for move, opp_as in opponent_as:
                if opp_as.game_won():
                    return move

        # Fork & Block Fork
        for move, afterstate in afterstates:
            if is_fork_move(afterstate.board, board.board_size, self.color, other_color):
                return move

        # Block Fork
        for move, afterstate in afterstates_opponent:
            if is_fork_move(afterstate.board, board.board_size, other_color, self.color):
                return move

        # First move: Randomly chose center or corner
        if (board.board == config.EMPTY).all():
            return choice(CORNERS + [(1, 1)])

        # Center
        if board.board[1][1] == config.EMPTY:
            return 1, 1

        # Opposite Corner
        for corner in CORNERS:
            if board.board[corner[0]][corner[1]] == other_color:
                return (corner[0]+2)%2, (corner[1]+2)%2

        # Empty Corner
        for corner in CORNERS:
            if board.board[corner[0]][corner[1]] == config.EMPTY:
                return corner

        # Empty Side
        for side in SIDES:
            if board.board[side[0]][side[1]] == config.EMPTY:
                return side


@njit
def is_fork_move(board, board_size, color, other_color):
    # check if two directions have a win chance: 3+3+1+1 possibilities
    win_directions = 0

    # lines
    for col in range(board_size):
        player_hor, opponent_hor = 0, 0
        player_ver, opponent_ver = 0, 0

        for row in range(board_size):
            # horizontal
            if board[col][row] == color:
                player_hor += 1
            if board[col][row] == other_color:
                opponent_hor += 1

            # vertical
            if board[row][col] == color:
                player_ver += 1
            if board[row][col] == other_color:
                opponent_ver += 1

        if (player_hor - opponent_hor) == 2:  # win direction
            win_directions += 1
        if (player_ver - opponent_ver) == 2:
            win_directions += 1

    player_diag, opponent_diag = 0, 0
    for pos in range(board_size):
        if board[pos][pos] == color:
            player_diag += 1
        if board[pos][pos] == other_color:
            opponent_diag += 1

    player_rev_diag, opponent_rev_diag = 0, 0
    for pos in range(board_size-1, -1, -1):
        if board[pos][pos] == color:
            player_rev_diag += 1
        if board[pos][pos] == other_color:
            opponent_rev_diag += 1

    if (player_diag - opponent_diag) == 2:
        win_directions += 1
    if (player_rev_diag - opponent_rev_diag) == 2:
        win_directions += 1

    return True if win_directions > 1 else False
