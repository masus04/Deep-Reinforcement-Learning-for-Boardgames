import numpy as np
from copy import deepcopy
import torch
from torch.distributions import Categorical

import TicTacToe.config as config
import abstractClasses as abstract
from TicTacToe.players.models import FCPolicyModel, LargeFCPolicyModel, ConvPolicyModel
from abstractClasses import PlayerException


class PPOPlayer(abstract.LearningPlayer):

    def __init__(self, strategy, lr, batch_size=1):
        super(PPOPlayer, self).__init__()

        if issubclass(strategy, abstract.Strategy):
            self.strategy = strategy(lr=lr, batch_size=1)  # Hard code batch size to 1 for now
        elif issubclass(strategy.__class__, abstract.Strategy):
            self.strategy = strategy
        else:
            raise Exception("ReinforcePlayer takes as a strategy argument a subclass of %s, received %s" % (abstract.Model, strategy))

    def get_move(self, board):
        representation, legal_moves = board.get_representation(self.color), board.get_legal_moves_map(self.color)
        if self.strategy.train:
            self.strategy.rewards.append(0)
            self.strategy.boards.append(representation)
            self.strategy.legal_move_maps.append(legal_moves)
        return self.strategy.evaluate(representation, legal_moves)

    def register_winner(self, winner_color):
        label = self.get_label(winner_color)
        if self.strategy.train:
            self.strategy.rewards[-1] = label
            return self.strategy.update()
        return 0


class PPOStrategy(abstract.Strategy):

    def __init__(self, lr, batch_size, gamma=config.GAMMA,  model=None):
        super(PPOStrategy, self).__init__()
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.model = model if model else FCPolicyModel()  # PGFCModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.steps = 5
        self.clip = 0.2

    def evaluate(self, board_sample, legal_moves_map):
        input = config.make_variable(torch.FloatTensor([board_sample]))
        legal_moves_map = config.make_variable(legal_moves_map)
        probs = self.model(input, legal_moves_map)

        distribution = Categorical(probs)
        action = distribution.sample()

        move = (action.data[0] // config.BOARD_SIZE, action.data[0] % config.BOARD_SIZE)
        log_prob = distribution.log_prob(action)
        if self.train:
            self.log_probs.append(log_prob)
        return move

    def update(self):
        # old_strategy = self.copy(shared_weights=False)
        self.rewards = self.discount_rewards(self.rewards, config.GAMMA)
        old_log_probs = [lp.data[0] for lp in self.log_probs]  # unpack so the original weights are not changed
        policy_loss = []

        for step in range(self.steps):
            for log_prob, old_log_prob, reward in zip(self.log_probs, old_log_probs, self.rewards):
                old_log_prob = config.make_variable(np.array([old_log_prob]))
                ratio = (log_prob - old_log_prob).exp()
                surr1 = ratio * reward
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * reward
                action_loss = -torch.min(surr1, surr2).mean()
                policy_loss.append(action_loss)
                action_loss.backward()

            del self.log_probs[:]

            if step+1 < self.steps:  # Prepare for next update
                for board, legal_move_map in zip(self.boards, self.legal_move_maps):
                    self.evaluate(board, legal_move_map)

        policy_loss = abs(torch.cat(policy_loss).mean().data[0])

        del self.rewards[:]
        del self.log_probs[:]
        del self.boards[:]
        del self.legal_move_maps[:]

        return policy_loss

    @staticmethod
    def normalize_rewards(rewards):
        return (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps)

    def copy(self, shared_weights=True):
        if shared_weights:
            strategy = self.__class__(model=self.model, lr=self.lr, batch_size=self.batch_size)
        else:
            strategy = self.__class__(model=self.model.copy(), lr=self.lr, batch_size=self.batch_size)

        strategy.train = deepcopy(self.train)
        return strategy


class FCPPOPlayer(PPOPlayer):
    def __init__(self, lr, strategy=None, batch_size=1):
        super(FCPPOPlayer, self).__init__(lr=lr, strategy=strategy if strategy is not None
                                                else PPOStrategy(lr, batch_size, model=LargeFCPolicyModel()))


class ConvPPOPlayer(PPOPlayer):
    def __init__(self, lr, strategy=None, batch_size=1):
        super(ConvPPOPlayer, self).__init__(lr=lr, strategy=strategy if strategy is not None
                                                  else PPOStrategy(lr, batch_size, model=ConvPolicyModel()))
