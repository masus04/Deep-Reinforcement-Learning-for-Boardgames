import numpy as np
from numba import njit

import Othello.config as config
from abstractClasses import Board, BoardException
from Othello.config import BLACK, WHITE, EMPTY


class OthelloBoard(Board):
    """
    Represents a board of Othello and all actions that can be taken on it.
    """
    def __init__(self, board=None):
        self.DIRECTIONS = [np.array([-1,-1]), np.array([-1,0]), np.array([-1,1]),
                           np.array([0, -1]), np.array([0, 0]), np.array([0, 1]),
                           np.array([1, -1]), np.array([1, 0]), np.array([1, 1])]

        self.board_size = board.board_size if board else config.BOARD_SIZE

        if board:
            self.board = board.board.copy()
        else:
            self.board = np.full((self.board_size, self.board_size), EMPTY, dtype=np.float64)
            self.board[3, 3] = config.WHITE
            self.board[3, 4] = config.BLACK
            self.board[4, 3] = config.BLACK
            self.board[4, 4] = config.WHITE

        self.illegal_move = None

    def get_valid_moves(self, color):
        other_color = self.other_color(color)
        legal_moves = set()
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == EMPTY:
                    pos = np.array([i, j])
                    for direction in self.DIRECTIONS:
                        new_pos = pos + direction
                        if self.in_bounds(new_pos) and self.board[new_pos[0], new_pos[1]] == other_color:
                            if self.get_legal_moves_in_direction(new_pos, direction, color, other_color):
                                legal_moves.add((pos[0], pos[1]))
        return legal_moves

    def get_legal_moves_in_direction(self, pos, direction, color, other_color):
        new_pos = pos + direction

        if not self.in_bounds(new_pos) or self.board[new_pos[0], new_pos[1]] == EMPTY:
            return False

        if self.board[new_pos[0], new_pos[1]] == color:
            return True

        if self.board[new_pos[0], new_pos[1]] == other_color:
            return self.get_legal_moves_in_direction(new_pos, direction, color, other_color)

    def apply_move(self, move, color):
        if color is None:
            raise BoardException("Illegal color provided: %s" % color)


        takes = set()
        takes.add(move)
        for direction in self.DIRECTIONS:
            takes = takes | self.apply_move_recursively(move+direction, direction, color, self.other_color(color), set())

        if len(takes) > 1:  # More than just placed stone in taken set
            for t in takes:
                self.board[t[0], t[1]] = color

        else:
            print("!! Illegal move !!")
            self.illegal_move = color

        return self

    def apply_move_recursively(self, pos, direction, color, other_color, dir_takes):
        if self.board[pos[0], pos[1]] == config.EMPTY:
            return set()

        if self.board[pos[0], pos[1]] == color:
            return dir_takes

        if self.board[pos[0], pos[1]] == other_color:
            dir_takes.add((pos[0], pos[1]))
            return self.apply_move_recursively(pos+direction, direction, color, other_color, dir_takes)

    def game_won(self):
        return len(self.get_valid_moves(config.BLACK) + self.get_valid_moves(config.WHITE)) == 0

    def in_bounds(self, position):
        return position[0] >= 0 and position[1] >= 0 and position[0] < self.board_size and position[1] < self.board_size

    def get_representation(self, color):
        if color == BLACK:
            return self.board.copy()

        if color == WHITE:
            return __get_representation_njit__(self.board, self.board_size, color, self.other_color(color))
        else:
            raise BoardException("Illegal color provided: %s" % color)

    def get_legal_moves_map(self, color):
        return __get_legal_moves_map__(self.board_size, self.get_valid_moves(color))

    def copy(self):
        return OthelloBoard(self)

    def count_stones(self):
        return __count_stones__(self.board, self.board_size)


"""   ---  Numba implementations  ---   '''
Numba 0.36 does not yet fully support custom types.
The following functions are implemented in such a way that they can be expressed using only supported types.
These functions can then be wrapped by a class method, hiding the numba implementation from the class user.
"""

@njit
def __get_legal_moves_map__(board_size, valid_moves):
    legal_moves_map = np.zeros((board_size, board_size))
    for move in valid_moves:
        legal_moves_map[move[0]][move[1]] = 1
    return legal_moves_map


@njit
def __count_stones__(board, board_size):
    """ returns a tuple (num_black_stones, num_white_stones)"""

    black = (board == np.full((board_size, board_size), BLACK, dtype=np.float64)).sum()
    white = (board == np.full((board_size, board_size), WHITE, dtype=np.float64)).sum()

    return black, white


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
