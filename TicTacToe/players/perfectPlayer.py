import os
import torch

import TicTacToe.config as config
from plotting import Printer
from TicTacToe.environment.board import TicTacToeBoard
from TicTacToe.players.searchPlayer import SearchPlayer
from abstractClasses import Player, PlayerException


class PerfectPlayer(Player):

    def __init__(self):
        super(PerfectPlayer, self).__init__()
        self.move_table = None

    def get_move(self, board):
        assert self.color

        if self.move_table is None:
            self.create_table()

        depth = sum(board.count_stones())
        for state, move in self.move_table[depth]:
            if (board.get_representation(self.color) == state.get_representation(self.color)).all():
                return move

    def create_table(self):
        # Zero depth
        search_player = SearchPlayer()
        search_player.color = config.BLACK
        temp_color = TicTacToeBoard.other_color(search_player.color)

        empty_board = set([TicTacToeBoard()])
        self.move_table = [[(a, search_player.get_move(a)) for a in empty_board]]

        for i in range(8):  # 8 moves
            search_player.color = temp_color
            temp_color = TicTacToeBoard.other_color(temp_color)

            temp_as = set()
            for afterstate, move in self.move_table[-1]:
                temp_as.update(a[0] for a in afterstate.get_afterstates(temp_color))

            self.move_table.append([(afterstate, search_player.get_move(afterstate)) for afterstate in temp_as])
            Printer.print_inplace("Created table up to depth: %s" % (i+2), round((i+2)/9*100))
