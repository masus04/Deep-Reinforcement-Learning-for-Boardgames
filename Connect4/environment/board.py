import numpy as np
from numba import njit

import Connect4.config as config
from abstractClasses import Board, BoardException
from Connect4.config import BLACK, WHITE, EMPTY

DIRECTIONS = np.array([[0, 1], [1, -1], [1, 0], [1, 1]])  # Check bottom right (dynamic programming)


class OthelloBoard(Board):
    """
    Represents a board of Othello and all actions that can be taken on it.
    """
    def __init__(self, board=None):
        self.board_size = board.board_size if board else config.BOARD_SIZE

        if board:
            self.board = board.board.copy()
        else:
            self.board = np.full((self.board_size[0], self.board_size[1]), EMPTY, dtype=np.float64)

        self.legal_moves = {}
        self.illegal_move = None

    def get_valid_moves(self, color):
        if color in self.legal_moves:
            return self.legal_moves[color]
        else:
            self.legal_moves[color] = __get_legal_moves__(self.board, self.board_size)
            return self.legal_moves[color]

    def apply_move(self, move, color):
        if color is None:
            raise BoardException("Illegal color provided: %s" % color)

        if move in self.legal_moves:  # More than just placed stone in taken set
            self.board[move[0], move[1]] = color
        else:
            print("!! Illegal move !!")
            self.illegal_move = color

        self.legal_moves = {}
        return self

    def game_won(self):
        # TODO: Adapt for Connect4
        if len(self.get_valid_moves(config.BLACK) | self.get_valid_moves(config.WHITE)) == 0:
            stones = self.count_connections()
            return config.BLACK if stones[0] > stones[1] else config.WHITE if stones[0] < stones[1] else config.EMPTY
        else:
            return None

    def get_representation(self, color):
        if color == BLACK:
            return self.board.copy()

        if color == WHITE:
            return __get_representation_njit__(self.board, self.board_size, color, self.other_color(color))
        else:
            raise BoardException("Illegal color provided: %s" % color)

    def get_legal_moves_map(self, color):
        legal_moves = self.get_valid_moves(color)
        if not legal_moves:
            return np.zeros((self.board_size, self.board_size))
        else:
            return __get_legal_moves_map__(self.board_size, legal_moves)

    def copy(self):
        return OthelloBoard(self)

    def count_connections(self):
        return count_connections(self.board, self.board_size)


"""   ---  Numba implementations  ---   '''
Numba 0.36 does not yet fully support custom types.
The following functions are implemented in such a way that they can be expressed using only supported types.
These functions can then be wrapped by a class method, hiding the numba implementation from the class user.
"""


# @njit
def __get_legal_moves__(board, board_size):
    legal_moves = set()
    for column in range(board_size[1]):
        row = 0
        while in_bounds(board_size, np.array([row, column])) and board[row, column] == EMPTY:  # Search highest occupied tile
            row += 1
        if row > 0:  # If column is not full, add lowest empty tile
            legal_moves.add((row-1, column))

    return legal_moves

@njit
def __get_legal_moves_map__(board_size, valid_moves):
    legal_moves_map = np.zeros((board_size, board_size))
    for move in valid_moves:
        legal_moves_map[move[0]][move[1]] = 1
    return legal_moves_map


# @njit
def count_connections(board, board_size):
    # TODO: Adapt for Connect4 -> Change to return longest connected rows (eg. 2x row of 3)
    """
    Calculates the longest connections and their number for both players

    :return a dictionary of the form {color: (connection_length, number_of_connections)}
    """

    connections = {config.BLACK: (1, 0), config.WHITE: (1, 0)}

    for col in range(board_size[1]):
        for row in range(board_size[0]):
            if board[row, col] != EMPTY:
                for d in DIRECTIONS:

                    color = board[row, col]
                    length = 0
                    new_pos = np.array([row, col])
                    while in_bounds(new_pos) and board[new_pos[0], new_pos[1]] == color:
                        length += 1
                        new_pos += d

                    if length > 1 and length == connections[color][0]:  # Connection of same length
                        connections[color][1] += 1
                    elif length > connections[color][0]:  # Connection of greater length
                        connections[color] = (length, 1)

    return connections


@njit
def __get_representation_njit__(board, board_size, color, other_color):
    out = np.full((board_size, board_size), config.EMPTY, dtype=np.float64)

    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == color:
                out[i,j] = other_color
            elif board[i, j] == other_color:
                out[i, j] = color

    return out


@njit
def in_bounds(board_size, position):
    return position[0] >= 0 and position[1] >= 0 and position[0] < board_size[0] and position[1] < board_size[1]


