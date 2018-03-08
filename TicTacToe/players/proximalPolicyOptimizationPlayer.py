import numpy as np
from copy import deepcopy
import torch
from torch.distributions import Categorical

import TicTacToe.config as config
import abstractClasses as abstract
from TicTacToe.players.models import FCPolicyModel, LargeFCPolicyModel, ConvPolicyModel
from abstractClasses import PlayerException


class PPOPlayer(abstract.LearningPlayer):

    def __init__(self, strategy, lr, clip):
        super(PPOPlayer, self).__init__()

        if issubclass(strategy, abstract.Strategy):
            self.strategy = strategy(lr=lr, clip=clip)
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

    def copy(self, shared_weights=True):
        return self.__class__(lr=self.strategy.lr, clip=self.strategy.clip, strategy=self.strategy.copy(shared_weights=shared_weights))


class PPOStrategy(abstract.Strategy):

    def __init__(self, lr, clip, gamma=config.GAMMA, model=None):
        super(PPOStrategy, self).__init__()
        self.lr = lr
        self.clip = clip
        self.gamma = gamma
        self.model = model if model else FCPolicyModel()  # PGFCModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.steps = 32

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

        for step in range(self.steps):
            policy_loss = []
            for log_prob, old_log_prob, reward in zip(self.log_probs, old_log_probs, self.rewards):
                old_log_prob = config.make_variable(np.array([old_log_prob]))
                ratio = (log_prob - old_log_prob).exp()
                surr1 = ratio * reward
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * reward
                action_loss = -torch.min(surr1, surr2).mean()
                policy_loss.append(action_loss)

                """
                self.optimizer.zero_grad()
                action_loss.backward()
                self.optimizer.step()
                """

            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).mean()
            policy_loss.backward()
            self.optimizer.step()

            del self.log_probs[:]

            if step+1 < self.steps:  # Prepare for next update
                for board, legal_move_map in zip(self.boards, self.legal_move_maps):
                    self.evaluate(board, legal_move_map)

        del self.rewards[:]
        del self.log_probs[:]
        del self.boards[:]
        del self.legal_move_maps[:]

        # policy_loss = (sum(policy_loss)/len(policy_loss)).data[0]
        return policy_loss

    @staticmethod
    def normalize_rewards(rewards):
        return (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps)

    def copy(self, shared_weights=True):
        if shared_weights:
            strategy = self.__class__(model=self.model, lr=self.lr, clip=self.clip)
        else:
            strategy = self.__class__(model=self.model.copy(), lr=self.lr, clip=self.clip)

        strategy.train = deepcopy(self.train)
        strategy.clip = deepcopy(self.clip)
        return strategy


class FCPPOPlayer(PPOPlayer):
    def __init__(self, lr, clip, strategy=None):
        super(FCPPOPlayer, self).__init__(lr=lr, clip=clip, strategy=strategy if strategy is not None
                                                else PPOStrategy(lr=lr, clip=clip, model=LargeFCPolicyModel()))


class ConvPPOPlayer(PPOPlayer):
    def __init__(self, lr, clip, strategy=None):
        super(ConvPPOPlayer, self).__init__(lr=lr, clip=clip, strategy=strategy if strategy is not None
                                                  else PPOStrategy(lr=lr, clip=clip, model=ConvPolicyModel()))
