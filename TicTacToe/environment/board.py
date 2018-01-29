import numpy as np

from TicTacToe.abstract_classes import Board
import TicTacToe.config as config
from TicTacToe.config import BLACK, WHITE, EMPTY


class TicTacToeBoard(Board):

    def __init__(self, board=None):
        self.board_size = board.board_size if board else config.BOARD_SIZE
        self.board = board.board.copy() if board else np.full((self.board_size, self.board_size), EMPTY, dtype=np.float64)

    def get_valid_moves(self, color=None):
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == EMPTY:
                    legal_moves.append((i, j))

        return legal_moves

    def apply_move(self, move, color):
        if color is None:
            raise BoardException("Illegal color provided: %s" % color)

        if move in self.get_valid_moves():
            self.board[move[0]][move[1]] = color
            return self
        else:
            raise BoardException("%s applied an illegal move: %s for board: \n%s" % (config.get_color_from_player_number(color), move, self.board))

    def game_won(self):
        if not self.get_valid_moves():
            return EMPTY

        for i in range(self.board_size):
            for j in range(self.board_size):
                # Make use of symmetry, only check bottom right half of directions
                for direction in [[1, 0], [-1, 1], [0, 1], [1, 1]]:
                    if self.__recursive_game_won__((i, j), direction, BLACK, 0):
                        return BLACK
                    if self.__recursive_game_won__((i, j), direction, WHITE, 0):
                        return WHITE
        return False

    def __recursive_game_won__(self, position, direction, color, depth):
        if depth >= config.WIN_LINE_LENGTH:
            return True
        if self.in_bounds(position) and self.board[position[0]][position[1]] == color:
            next_p = (position[0] + direction[0], position[1] + direction[1])
            return self.__recursive_game_won__(next_p, direction, color, depth+1)
        return False

    def in_bounds(self, position):
        for p in position:
            if not (p >= 0 and p < self.board_size):
                return False
        return True


    def get_representation(self, color):
        if color == BLACK:
            return self.board.copy()

        if color == WHITE:
            representation = []
            for row in self.board:
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

    def get_legal_moves_map(self, color):
        legal_moves_map = np.zeros([self.board_size, self.board_size])
        for move in self.get_valid_moves(color):
            legal_moves_map[move[0]][move[1]] = 1

    def copy(self):
        return TicTacToeBoard(self)

    @staticmethod
    def other_color(color):
        if color == BLACK:
            return WHITE
        if color == WHITE:
            return BLACK
        if color == EMPTY:
            return color
        raise BoardException("Illegal color provided: %s" % color)

    def count_stones(self):
        """ returns a tuple (num_black_stones, num_white_stones)"""
        black = 0
        white = 0
        for col in self.board:
            for tile in col:
                if tile == config.BLACK:
                    black += 1
                elif tile == config.WHITE:
                    white += 1
        return black, white


class BoardException(Exception):
    pass
