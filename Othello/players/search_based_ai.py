import sys
import random
from queue import PriorityQueue
from functools import lru_cache


class GameArtificialIntelligence(object):

    def __init__(self, heuristic_fn):
        self.heuristic = heuristic_fn

    @lru_cache(maxsize=2**10)
    def move_search(self, starting_node, depth, current_player, other_player):
        self.player = current_player
        self.other_player = other_player
        possible_moves = starting_node.get_valid_moves(current_player)
        if len(possible_moves) == 1:
            return list(possible_moves)[0]

        score = -sys.maxsize
        move = None
        self.queue = PriorityQueue(len(possible_moves))

        (new_move, new_score) = self.alpha_beta_wrapper(starting_node, depth, current_player, other_player)
        if new_move is not None:
            move = new_move
            score = new_score
            # print "Got to Depth:", depth
        return move


    def alpha_beta_wrapper(self, node, depth, current_player, other_player):
        alpha = -sys.maxsize-1
        beta = sys.maxsize
        if self.queue.queue:
            children = self.queue.queue
            self.queue = PriorityQueue(self.queue.maxsize)
            for (x, child, move) in children:
                new_alpha = self.alpha_beta_search(child, depth-1, other_player, current_player, alpha, beta, False)
                if new_alpha is None:
                    return (None, None)
                else:
                    self.queue.put((-new_alpha, child, move))
                if new_alpha > alpha:
                    alpha = new_alpha
                    best_move = move
                #print "Possible move:", move, "Score:", new_alpha
        else:
            children = node.child_nodes(current_player)
            # Shuffle order of moves evaluated to prevent playing the same game every time
            random.shuffle(children)
            for (child, move) in children:
                new_alpha = self.alpha_beta_search(child, depth-1, other_player, current_player, alpha, beta, False)
                if new_alpha is None:
                    return (None, None)
                else:
                    self.queue.put((-new_alpha, child, move))
                if new_alpha > alpha:
                    alpha = new_alpha
                    best_move = move
                #print "Possible move:", move, "Score:", new_alpha
        return (best_move, alpha)

    def keyify(self, node, player):
        from hashlib import sha1
        return sha1(node.board.data).hexdigest()

    def alpha_beta_search(self, node, depth, current_player, other_player, alpha=-sys.maxsize-1, beta=sys.maxsize, maximizing=True):
        if depth == 0 or node.game_won() is not None:
            return self.heuristic(node, self.player, self.other_player)

        children = node.child_nodes(current_player)
        if maximizing:
            if len(children) == 0:
                new_alpha = self.alpha_beta_search(node, depth-1, other_player, current_player, alpha, beta, False)
                if new_alpha is None:
                    return None
                alpha = max(alpha, new_alpha)
            else:
                for (child, move) in children:
                    new_alpha = self.alpha_beta_search(child, depth-1, other_player, current_player, alpha, beta, False)
                    if new_alpha is None:
                        return None
                    alpha = max(alpha, new_alpha)
                    if alpha >= beta:
                        break
            return alpha
        else:
            if len(children) == 0:
                new_beta = self.alpha_beta_search(node, depth-1, other_player, current_player, alpha, beta)
                if new_beta is None:
                    return None
                beta = min(beta, new_beta)
            else:
                for (child, move) in children:
                    new_beta = self.alpha_beta_search(child, depth-1, other_player, current_player, alpha, beta)
                    if new_beta is None:
                        return None
                    beta = min(beta, new_beta)
                    if beta <= alpha:
                        break
            return beta
