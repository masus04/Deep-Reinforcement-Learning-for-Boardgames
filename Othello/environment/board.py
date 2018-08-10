import numpy as np
from numba import njit

import Othello.config as config
from abstractClasses import Board, BoardException
from Othello.config import BLACK, WHITE, EMPTY

DIRECTIONS = np.array([[-1,-1], [-1,0], [-1,1],
                       [0, -1], [0, 0], [0, 1],
                       [1, -1], [1, 0], [1, 1]])


class OthelloBoard(Board):
    """
    Represents a board of Othello and all actions that can be taken on it.
    """
    def __init__(self, board=None):
        self.board_size = board.board_size if board else config.BOARD_SIZE

        if board:
            self.board = board.board.copy()
        else:
            self.board = np.full((self.board_size, self.board_size), EMPTY, dtype=np.float64)
            self.board[3, 3] = config.WHITE
            self.board[3, 4] = config.BLACK
            self.board[4, 3] = config.BLACK
            self.board[4, 4] = config.WHITE

        self.legal_moves = {}
        self.illegal_move = None

    def get_valid_moves(self, color):
        if color in self.legal_moves:
            return self.legal_moves[color]
        else:
            self.legal_moves[color] = __get_legal_moves__(self.board, self.board_size, color, self.other_color(color))
            return self.legal_moves[color]

    def apply_move(self, move, color):
        if color is None:
            raise BoardException("Illegal color provided: %s" % color)

        takes = find_takes(self.board, self.board_size, move, color, self.other_color(color))

        if len(takes) > 1:  # More than just placed stone in taken set
            for t in takes:
                self.board[t[0], t[1]] = color

        else:
            print("!! Illegal move !!")
            self.illegal_move = color

        self.legal_moves = {}
        return self

    def game_won(self):
        if len(self.get_valid_moves(config.BLACK) | self.get_valid_moves(config.WHITE)) == 0:
            stones = self.count_stones()
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

    def child_nodes(self, color):
        return [(self.copy().apply_move(move=move, color=color), move) for move in self.get_valid_moves(color)]

    def count_stones(self):
        return __count_stones__(self.board, self.board_size)

    def get_empty_spaces(self):
        stone_count = self.count_stones()
        return self.board_size**2 - stone_count[0] - stone_count[1]

    def __eq__(self, other):
        return (self.board == other.board).all()

    def __hash__(self):
        return hash(str(self.board))


"""   ---  Numba implementations  ---   '''
Numba 0.36 does not yet fully support custom types.
The following functions are implemented in such a way that they can be expressed using only supported types.
These functions can then be wrapped by a class method, hiding the numba implementation from the class user.
"""


@njit
def __get_legal_moves__(board, board_size, color, other_color):
    legal_moves = set()
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == EMPTY:
                pos = np.array([i, j])
                legal = False  # Optimization preventing unnecessary checks after move has been proven valid
                for k in range(len(DIRECTIONS)):
                    direction = DIRECTIONS[k]
                    new_pos = pos + direction
                    if not legal and in_bounds(board_size, new_pos) and board[new_pos[0], new_pos[1]] == other_color:
                        if __get_legal_moves_in_direction__(board, board_size, new_pos, direction, color, other_color):
                            legal_moves.add((pos[0], pos[1]))
                            legal = True
    return legal_moves


@njit
def __get_legal_moves_in_direction__(board, board_size, pos, direction, color, other_color):
    """
    Checks for player stones in the given direction AFTER the given position that mark the original position as a legal move

    :param board: the board
    :param board_size: the board size (NxN)
    :param pos: The position of the first oppoonent stone
    :param direction: direction to check
    :param color: player color
    :param other_color: opponent color
    :return: True if at least one stone can be overturned in the given direction and False otherwise
    """
    new_pos = pos + direction

    while in_bounds(board_size, new_pos) and board[new_pos[0], new_pos[1]] != EMPTY:

        if board[new_pos[0], new_pos[1]] == color:
            return True

        if board[new_pos[0], new_pos[1]] == other_color:
            new_pos = new_pos + direction

    return False


@njit
def __get_legal_moves_map__(board_size, valid_moves):
    legal_moves_map = np.zeros((board_size, board_size))
    for move in valid_moves:
        legal_moves_map[move[0]][move[1]] = 1
    return legal_moves_map


@njit
def find_takes(board, board_size, move, color, other_color):
    takes = [move]
    for i in range(len(DIRECTIONS)):
        direction = DIRECTIONS[i]
        new_pos = np.array(move) + direction
        if in_bounds(board_size, new_pos) and board[new_pos[0], new_pos[1]] == other_color:
            find_takes_directionally(board, board_size, new_pos, direction, color, other_color, takes)
    return takes


@njit
def find_takes_directionally(board, board_size, pos, direction, color, other_color, takes):
    dir_takes = [(pos[0], pos[1])]

    new_pos = pos + direction
    while in_bounds(board_size, (new_pos[0], new_pos[1])) and board[new_pos[0], new_pos[1]] != config.EMPTY:

        if board[new_pos[0], new_pos[1]] == color:
            takes.extend(dir_takes)
            return

        if board[new_pos[0], new_pos[1]] == other_color:
            dir_takes.append((new_pos[0], new_pos[1]))
            new_pos += direction

    return


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


@njit
def in_bounds(board_size, position):
    return position[0] >= 0 and position[1] >= 0 and position[0] < board_size and position[1] < board_size
