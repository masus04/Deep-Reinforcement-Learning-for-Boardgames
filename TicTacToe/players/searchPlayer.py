from abstractClasses import Player

import TicTacToe.config as config
from TicTacToe.environment.board import Board
from TicTacToe.players.search_based_ai import GameArtificialIntelligence


class SearchPlayer(Player):

    def __init__(self, search_depth=10):
        super(SearchPlayer, self).__init__()
        self.search_depth = search_depth
        self.ai = GameArtificialIntelligence(evaluate)

    def get_move(self, board):
        assert self.color
        return self.ai.move_search(board, self.search_depth, self.color, board.other_color(self.color))

    def __str__(self):
        return "[%s search depth %s]" % (self.__class__.__name__, self.search_depth)


def evaluate(board, current_player, other_player):

    if board.game_won() == current_player:
        return 100  # - (9-sum(board.count_stones()))

    if board.game_won() == config.EMPTY:
        return 50

    if board.game_won() == other_player:
        return - 10  # - (9-sum(board.count_stones()))

    return 0
