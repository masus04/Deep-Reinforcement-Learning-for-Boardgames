import sys
from Othello.config import BLACK, WHITE, EMPTY


class OthelloHeuristic(object):

    WIN = sys.maxsize - 1000

    START_GAME = 0
    MID_GAME = 1
    END_GAME = 2

    DEFAULT_STRATEGY = 0
    PURE_SAVE_STONES_STRATEGY = 1
    PURE_MOBILITY_STRATEGY = 2
    GREEDY_STRATEGY = 3

    def __init__(self, strategy=DEFAULT_STRATEGY):
        if strategy == self.DEFAULT_STRATEGY:
            self.PIECE_COUNT_FACTOR = [0, 0, 1]
            self.CORNER_FACTOR = [1000, 1000, 0]
            self.MOBILITY_FACTOR = [250, 300, 0]
            self.EDGE_FACTOR = [25, 25, 0]
            self.CORNER_EDGE_FACTOR = [400, 400, 0]
            self.STABILITY_FACTOR = [120, 120, 0]
            self.SAVE_STONES_FACTOR = [1000, 1000, 0]
        else:
            self.PIECE_COUNT_FACTOR = [0, 0, 1]
            self.CORNER_FACTOR = [0, 0, 0]
            self.MOBILITY_FACTOR = [0, 0, 0]
            self.EDGE_FACTOR = [0, 0, 0]
            self.CORNER_EDGE_FACTOR = [0, 0, 0]
            self.STABILITY_FACTOR = [0, 0, 0]
            self.SAVE_STONES_FACTOR = [0, 0, 0]

        if strategy == self.PURE_SAVE_STONES_STRATEGY:
            self.SAVE_STONES_FACTOR = [1, 1, 0]

        if strategy == self.PURE_MOBILITY_STRATEGY:
            self.MOBILITY_FACTOR = [1, 1, 0]

        if strategy == self.GREEDY_STRATEGY:
            self.PIECE_COUNT_FACTOR = [1, 1, 1]

    def evaluate(self, board, current_player, other_player):
        # Check for win conditions
        winner = board.game_won()
        if winner is not None:
            if winner == current_player:
                return OthelloHeuristic.WIN + self.evaluate_piece_count(board, current_player, other_player, OthelloHeuristic.END_GAME)
            elif winner == other_player:
                return -OthelloHeuristic.WIN-1 + self.evaluate_piece_count(board, current_player, other_player, OthelloHeuristic.END_GAME)

        # Determine Game State to Determine Heuristic Values
        if board.get_empty_spaces() >= 45:
            game_state = OthelloHeuristic.START_GAME
        elif board.get_empty_spaces() >= 2:
            game_state = OthelloHeuristic.MID_GAME
        else:
            game_state = OthelloHeuristic.END_GAME

        score = 0
        if self.CORNER_FACTOR[game_state] != 0:
            score += self.evaluate_corner_pieces(board, current_player, other_player, game_state)
        if self.PIECE_COUNT_FACTOR[game_state] != 0:
            score += self.evaluate_piece_count(board, current_player, other_player, game_state)
        if self.MOBILITY_FACTOR[game_state] != 0:
            score += self.evaluate_mobility(board, current_player, other_player, game_state)
        if self.EDGE_FACTOR[game_state] != 0:
            score += self.evaluate_edge_pieces(board, current_player, other_player, game_state)
        if self.CORNER_EDGE_FACTOR[game_state] != 0:
            score += self.evaluate_corner_edge(board, current_player, other_player, game_state)
        if self.STABILITY_FACTOR[game_state] != 0:
            score += self.evaluate_stability(board, current_player, other_player, game_state)
        if self.SAVE_STONES_FACTOR[game_state] != 0:
            score += self.evaluate_save_stones(board, current_player, other_player, game_state)
        return score

    def evaluate_heuristical_strategy(self, board, current_player, other_player):
        """ Ebaluates the given board according to a given table of board values often used for comparison in research """
        values = [[100], [-25], [10], [5], [5], [10], [-25], [100]
                  [-25], [-25], [ 2], [2], [2], [ 2], [-25], [-25],
                  [ 10], [  2], [ 5], [1], [1], [ 5], [  2], [ 10],
                  [  5], [  2], [ 1], [2], [2], [ 1], [  2], [  5],
                  [  5], [  2], [ 1], [2], [2], [ 1], [  2], [  5],
                  [ 10], [  2], [ 5], [1], [1], [ 5], [  2], [ 10],
                  [-25], [-25], [ 2], [2], [2], [ 2], [-25], [-25],
                  [100], [-25], [10], [5], [5], [10], [-25], [100]]
        score = 0
        for i in range(len(board.board)):
            for j in range(len(board.board[0])):
                if board.board[i][j] == current_player:
                    score += values[i][j]
                elif board.board[i][j] == other_player:
                    score -= values[i][j]
        return score

    def evaluate_piece_count(self, board, current_player, other_player, game_state):
        stone_count = board.count_stones()
        score = 0
        if current_player == BLACK:
            score += self.PIECE_COUNT_FACTOR[game_state]*stone_count[0]
            score -= self.PIECE_COUNT_FACTOR[game_state]*stone_count[1]
        else:
            score += self.PIECE_COUNT_FACTOR[game_state] * stone_count[1]
            score -= self.PIECE_COUNT_FACTOR[game_state] * stone_count[0]
        return score

    def evaluate_corner_pieces(self, board, current_player, other_player, game_state):
        score = 0
        for i in [0, 7]:
            for j in [0, 7]:
                if board.board[i][j] == current_player:
                    score += self.CORNER_FACTOR[game_state]
                elif board.board[i][j] == other_player:
                    score -= self.CORNER_FACTOR[game_state]
        return score

    def evaluate_edge_pieces(self, board, current_player, other_player, game_state):
        score = 0
        # Compute Horizontal Edges
        for i in [0, 7]:
            for j in range(2, 6):
                if board.board[i][j] == current_player:
                    score += self.EDGE_FACTOR[game_state]
                elif board.board[i][j] == other_player:
                    score -= self.EDGE_FACTOR[game_state]
        # Comput Vertical Edges
        for i in range(2, 6):
            for j in [0, 7]:
                if board.board[i][j] == current_player:
                    score += self.EDGE_FACTOR[game_state]
                elif board.board[i][j] == other_player:
                    score -= self.EDGE_FACTOR[game_state]
        return score

    def evaluate_stability(self, board, current_player, other_player, game_state):
        """Calculates difference between both player's save edge stones but does not correct for redundancy when an entire edge is occupied by the same player. Also does not calculate other save stones"""
        score = 0
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for corner in corners:
            score += self.edge_stability(board, corner, current_player, game_state)
            score -= self.edge_stability(board, corner, other_player, game_state)
        return score

    def edge_stability(self, board, corner, current_player, game_state):
        """Calculates number of save edge stones for a given player"""
        score = 0
        if corner == (0, 0) and board.board[corner[0]][corner[1]] == current_player:
            score += self.STABILITY_FACTOR[game_state]
            i = 1
            while i < 7 and board.board[i][corner[1]] == current_player:
                score += self.STABILITY_FACTOR[game_state]
                i += 1
            i = 1
            while i < 7 and board.board[corner[0]][i] == current_player:
                score += self.STABILITY_FACTOR[game_state]
                i += 1
        elif corner == (0, 7) and board.board[corner[0]][corner[1]] == current_player:
            score += self.STABILITY_FACTOR[game_state]
            i = 1
            while i < 7 and board.board[i][corner[1]] == current_player:
                score += self.STABILITY_FACTOR[game_state]
                i += 1
            i = 6
            while i > 0 and board.board[corner[0]][i] == current_player:
                score += self.STABILITY_FACTOR[game_state]
                i -= 1
        elif corner == (7, 0) and board.board[corner[0]][corner[1]] == current_player:
            score += self.STABILITY_FACTOR[game_state]
            i = 6
            while i > 0 and board.board[i][corner[1]] == current_player:
                score += self.STABILITY_FACTOR[game_state]
                i -= 1
            i = 1
            while i < 7 and board.board[corner[0]][i] == current_player:
                score += self.STABILITY_FACTOR[game_state]
                i += 1
        elif corner == (7, 7) and board.board[corner[0]][corner[1]] == current_player:
            score += self.STABILITY_FACTOR[game_state]
            i = 6
            while i > 0 and board.board[i][corner[1]] == current_player:
                score += self.STABILITY_FACTOR[game_state]
                i -= 1
            i = 6
            while i > 0 and board.board[corner[0]][i] == current_player:
                score += self.STABILITY_FACTOR[game_state]
                i -= 1
        return score

    def evaluate_corner_edge(self, board, current_player, other_player, game_state):
        """Calculates the number of tiles surrounding an EMPTY corner for both players, scoring them negatively"""
        score = 0
        corner = (0, 0)
        for (i, j) in [(1, 0), (1, 1), (0, 1)]:
            if board.board[corner[0]][corner[1]] == EMPTY:
                if board.board[i][j] == current_player:
                    if i == 1 and j == 1:
                        score -= self.CORNER_EDGE_FACTOR[game_state]
                    else:
                        score -= self.CORNER_EDGE_FACTOR[game_state] / 2
                elif board.board[i][j] == other_player:
                    if i == 1 and j == 1:
                        score += self.CORNER_EDGE_FACTOR[game_state]
                    else:
                        score += self.CORNER_EDGE_FACTOR[game_state] / 2
        corner = (7, 0)
        for (i, j) in [(6, 0), (6, 1), (7, 1)]:
            if board.board[corner[0]][corner[1]] == EMPTY:
                if board.board[i][j] == current_player:
                    if i == 6 and j == 1:
                        score -= self.CORNER_EDGE_FACTOR[game_state]
                    else:
                        score -= self.CORNER_EDGE_FACTOR[game_state] / 2
                elif board.board[i][j] == other_player:
                    if i == 6 and j == 1:
                        score += self.CORNER_EDGE_FACTOR[game_state]
                    else:
                        score += self.CORNER_EDGE_FACTOR[game_state] / 2
        corner = (7, 7)
        for (i, j) in [(7, 6), (6, 6), (6, 7)]:
            if board.board[corner[0]][corner[1]] == EMPTY:
                if board.board[i][j] == current_player:
                    if i == 6 and j == 6:
                        score -= self.CORNER_EDGE_FACTOR[game_state]
                    else:
                        score -= self.CORNER_EDGE_FACTOR[game_state] / 2
                elif board.board[i][j] == other_player:
                    if i == 6 and j == 6:
                        score += self.CORNER_EDGE_FACTOR[game_state]
                    else:
                        score += self.CORNER_EDGE_FACTOR[game_state] / 2
        corner = (0, 7)
        for (i, j) in [(0, 6), (1, 6), (1, 7)]:
            if board.board[corner[0]][corner[1]] == EMPTY:
                if board.board[i][j] == current_player:
                    if i == 1 and j == 6:
                        score -= self.CORNER_EDGE_FACTOR[game_state]
                    else:
                        score -= self.CORNER_EDGE_FACTOR[game_state] / 2
                elif board.board[i][j] == other_player:
                    if i == 1 and j == 6:
                        score += self.CORNER_EDGE_FACTOR[game_state]
                    else:
                        score += self.CORNER_EDGE_FACTOR[game_state] / 2
        return score

    def evaluate_save_stones(self, board, current_player, other_player, game_state):
        return self.evaluate_save_stones_h(board, current_player, other_player, game_state) - self.evaluate_save_stones_h(board, other_player, current_player, game_state)

    def evaluate_save_stones_h(self, board, current_player, other_player, game_state):
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        all_save_stones = set()  # Use array instead?
        for corner, direction in zip(corners, directions):
            save_stones = []
            # Extend in y direction
            if board.board[corner[0]][corner[1]] == current_player:
                # invert axes in order to extend in y direction first
                self.extend_in_y(board, corner, direction, current_player, save_stones)

                # Extend in x direction by dinamic programming
                for tile in save_stones:
                    self.extend_save_stones(board, tile, direction, current_player, all_save_stones)

        return self.SAVE_STONES_FACTOR[game_state] * len(all_save_stones)

    def extend_save_stones(self, board, tile, direction, current_player, save_stones):
        """Extends save stones in x direction"""
        guardian1 = self.guarding((tile[0]-direction[0], tile[1]), save_stones)  # The stone that makes this one safe in the other axis
        guardian2 = self.guarding((tile[0]-direction[0], tile[1]+direction[1]), save_stones)
        guardian3 = self.guarding((tile[0]+direction[0], tile[1]-direction[1]), save_stones)

        guarded = guardian1 and (guardian2 or guardian3)
        if self.coordinates_in_bounds(tile) and board.board[tile[0]][tile[1]] == current_player and guarded:
            save_stones.add(tile)
            self.extend_save_stones(board, (tile[0], tile[1]+direction[1]), direction, current_player, save_stones)

    def extend_in_y(self, board, tile, direction, current_player, save_stones):
        if self.coordinates_in_bounds(tile) and board.board[tile[0]][tile[1]] == current_player:
            save_stones.append(tile)
            self.extend_in_y(board, (tile[0]+direction[0], tile[1]), direction, current_player, save_stones)

    @staticmethod
    def coordinates_in_bounds(coordinates):
        return coordinates[0] >= 0 and coordinates[0] < 8 and coordinates[1] >= 0 and coordinates[1] < 8

    def guarding(self, guardian, save_stones):
        return guardian in save_stones or not self.coordinates_in_bounds(guardian)

    def evaluate_mobility(self, board, current_player, other_player, game_state):
        score = 0
        score += len(board.get_valid_moves(current_player))*self.MOBILITY_FACTOR[game_state]
        score -= len(board.get_valid_moves(other_player))*self.MOBILITY_FACTOR[game_state]
        return score
