import datetime
import sys
import random
from queue import PriorityQueue


class GameArtificialIntelligence(object):

    def __init__(self, heuristic_fn):
        self.heuristic = heuristic_fn
        self.trans_table = dict()

    def move_search(self, starting_node, time_limit, current_player, other_player):
        self.player = current_player
        self.other_player = other_player
        possible_moves = starting_node.get_valid_moves(current_player)
        if len(possible_moves) == 1:
            # print("Only 1 Possible Move:", possible_moves[0])
            return possible_moves[0]
        depth = 0
        score = -sys.maxsize - 1
        move = None
        time_start = datetime.datetime.now()
        self.time_done = time_start + datetime.timedelta(seconds=time_limit)
        time_cutoff = time_start + datetime.timedelta(seconds=time_limit/2.0)
        self.cutoff = False
        WIN = sys.maxsize - 1000
        self.queue = PriorityQueue(len(possible_moves))
        self.first = True
        while datetime.datetime.now() < time_cutoff and not self.cutoff and starting_node.empty_spaces >= depth:
            depth += 1
            (new_move, new_score) = self.alpha_beta_wrapper(starting_node, depth, current_player, other_player)
            if new_move is not None and not self.cutoff:
                move = new_move
                score = new_score
                # print "Got to Depth:", depth
            else:
                # print "Cutoff at depth", depth
                pass
        return move

    def alpha_beta_wrapper(self, node, depth, current_player, other_player):
        alpha = -sys.maxsize-1
        beta = sys.maxsize
        if self.first:
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
            self.first = False
        else:
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
        return (best_move, alpha)

    def keyify(self, node, player):
        from hashlib import sha1
        return sha1(node.board.data).hexdigest()

    def alpha_beta_search(self, node, depth, current_player, other_player, alpha=-sys.maxsize-1, beta=sys.maxsize, maximizing=True):
        if datetime.datetime.now() > self.time_done - datetime.timedelta(milliseconds=10):
            self.cutoff = True
            return None
        if depth == 0 or node.game_won() is not None:
            key = self.keyify(node, self.player)
            if key in self.trans_table:
                return self.trans_table[key]
            val = self.heuristic(node, self.player, self.other_player)
            self.trans_table[key] = val
            return val
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

    def alpha_beta_search_trans(self, node, depth, current_player, other_player, alpha=-sys.maxsize-1, beta=sys.maxsize, maximizing=True):
        if datetime.datetime.now() > self.time_done - datetime.timedelta(milliseconds=10):
            self.cutoff = True
            return None
        if depth == 0 or node.game_won() is not None:
            return self.heuristic(node, self.player, self.other_player)
        children = node.child_nodes(current_player)
        if maximizing:
            if len(children) == 0:
                node.board.flags.writeable = False
                key = self.keyify(node, other_player)
                if key in self.trans_table and self.trans_table[key][0] >= depth-1:
                    new_alpha = self.trans_table[key][1]
                else:
                    new_alpha = self.alpha_beta_search(node, depth-1, other_player, current_player, alpha, beta, False)
                    if new_alpha is not None:
                        self.trans_table[key] = (depth-1, new_alpha)
                if new_alpha is None:
                    return None
                alpha = max(alpha, new_alpha)
            else:
                for (child, move) in children:
                    child.board.flags.writeable = False
                    key = self.keyify(child, other_player)
                    if key in self.trans_table and self.trans_table[key][0] >= depth-1:
                        new_alpha = self.trans_table[key][1]
                    else:
                        new_alpha = self.alpha_beta_search(child, depth-1, other_player, current_player, alpha, beta, False)
                        if new_alpha is not None:
                            self.trans_table[key] = (depth-1, new_alpha)
                    if new_alpha is None:
                        return None
                    alpha = max(alpha, new_alpha)
                    if alpha >= beta:
                        break
            return alpha
        else:
            if len(children) == 0:
                node.board.flags.writeable = False
                key = self.keyify(node, other_player)
                if key in self.trans_table and self.trans_table[key][0] >= depth-1:
                    new_beta = self.trans_table[key][1]
                else:
                    new_beta = self.alpha_beta_search(node, depth-1, other_player, current_player, alpha, beta)
                    if new_beta is not None:
                        self.trans_table[key] = (depth-1, new_beta)
                if new_beta is None:
                    return None
                beta = min(beta, new_beta)
            else:
                for (child, move) in children:
                    child.board.flags.writeable = False
                    key = self.keyify(child, other_player)
                    if key in self.trans_table and self.trans_table[key][0] >= depth-1:
                        new_beta = self.trans_table[key][1]
                    else:
                        new_beta = self.alpha_beta_search(child, depth-1, other_player, current_player, alpha, beta)
                        if new_beta is not None:
                            self.trans_table[key] = (depth-1, new_beta)
                    if new_beta is None:
                        return None
                    beta = min(beta, new_beta)
                    if beta <= alpha:
                        break
            return beta
