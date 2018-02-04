import numpy as np
from numba import njit, float64, intp, boolean

from abstractClasses import Board, BoardException
import TicTacToe.config as config
from TicTacToe.config import BLACK, WHITE, EMPTY


class TicTacToeBoard(Board):
    """
    Represents a board of TicTacToe and all actions that can be taken on it.
    """
    def __init__(self, board=None):
        self.board_size = board.board_size if board else config.BOARD_SIZE
        self.board = board.board.copy() if board else np.full((self.board_size, self.board_size), EMPTY, dtype=np.float64)
        self.illegal_move = None

    def get_valid_moves(self, color=None):
        return __get_valid_moves__(self.board, self.board_size)

    def apply_move(self, move, color):
        if color is None:
            raise BoardException("Illegal color provided: %s" % color)

        if move in self.get_valid_moves():
            self.board[move[0]][move[1]] = color
            return self
        else:
            self.illegal_move = color
        return self

    def game_won(self):
        if self.illegal_move is not None:
            return self.other_color(self.illegal_move)

        valid_moves = __get_valid_moves__(self.board, self.board_size)
        if not valid_moves:
            return EMPTY

        return __game_won__(self.board, self.board_size, valid_moves)

    def in_bounds(self, position):
        return __in_bounds__(position, self.board_size)

    def get_representation(self, color):
        return __get_representation__(self.board, color)

    def get_legal_moves_map(self, color):
        return __get_legal_moves_map__(self.board_size, __get_valid_moves__(self.board, self.board_size))

    def copy(self):
        return TicTacToeBoard(self)

    def count_stones(self):
        return __count_stones__(self.board, self.board_size)


"""   ---  Numba implementations  ---   '''
Numba 0.36 does not yet fully support custom types.
The following methods are implemented in such a way that they can be expressed using only supported types.
These methods can then be wrapped by a class method, hiding the numba implementation from the class user.
"""


@njit
def __get_valid_moves__(board, board_size):
    legal_moves = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == EMPTY:
                legal_moves.append((i, j))
    return legal_moves


@njit
def __game_won__(board, board_size, valid_moves):
    for i in range(board_size):
        for j in range(board_size):
            # Make use of symmetry, only check bottom right half of directions
            for direction in [(1, 0), (-1, 1), (0, 1), (1, 1)]:
                if __recursive_game_won__(board, board_size, (i, j), direction, BLACK, 0):
                    return BLACK
                if __recursive_game_won__(board, board_size, (i, j), direction, WHITE, 0):
                    return WHITE
    return False


@njit
def __recursive_game_won__(board, board_size, position, direction, color, depth):
    if depth >= config.WIN_LINE_LENGTH:
        return True
    if __in_bounds__(position, board_size) and board[position[0]][position[1]] == color:
        next_p = (position[0] + direction[0], position[1] + direction[1])
        return __recursive_game_won__(board, board_size, next_p, direction, color, depth+1)
    return False


@njit
def __in_bounds__(position, board_size):
    for p in position:
        if not (p >= 0 and p < board_size):
            return False
    return True


#@njit
def __get_representation__(board, color):
    if color == BLACK:
        return board.copy()

    if color == WHITE:
        representation = []
        for row in board:
            new_row = []
            for field in row:
                if field == EMPTY:
                    new_row.append(EMPTY)
                elif field == BLACK:
                    new_row.append(WHITE)
                else:
                    new_row.append(BLACK)
            representation.append(new_row)

        return np.array(representation, dtype=np.float64)

    raise BoardException("Illegal color provided: %s" % color)


@njit
def __get_legal_moves_map__(board_size, valid_moves):
    legal_moves_map = np.zeros([board_size, board_size])
    for move in valid_moves:
        legal_moves_map[move[0]][move[1]] = 1


@njit
def __count_stones__(board, board_size):
    """ returns a tuple (num_black_stones, num_white_stones)"""

    black = (board == np.full((board_size, board_size), BLACK, dtype=np.float64)).sum()
    white = (board == np.full((board_size, board_size), WHITE, dtype=np.float64)).sum()

    return black, white
