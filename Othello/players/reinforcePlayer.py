import torch
from torch.distributions import Categorical

import Othello.config as config
import abstractClasses as abstract
from abstractClasses import LearningPlayer, PlayerException
from models import FCPolicyModel, LargeFCPolicyModel, HugeFCPolicyModel, ConvPolicyModel


class PGStrategy(abstract.Strategy):

    def __init__(self, lr, weight_decay, gamma=config.GAMMA,  model=None):
        super(PGStrategy, self).__init__()
        self.lr = lr
        self.gamma = gamma
        self.weight_decay = weight_decay

        self.model = model if model else FCPolicyModel(config=config)  # PGFCModel()
        self.model.double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def evaluate(self, board_sample, legal_moves_map):
        input = config.make_variable([board_sample])

        probs, state_value = self.model(input, config.make_variable(legal_moves_map))
        distribution = Categorical(probs)
        action = distribution.sample()

        move = (int(action) // config.BOARD_SIZE, int(action) % config.BOARD_SIZE)
        if self.train:
            self.log_probs.append(distribution.log_prob(action))
        return move

    def update(self):
        if not self.train:
            return 0

        if len(self.log_probs) != len(self.rewards):
            raise PlayerException("log_probs length must be equal to rewards length. Got %s - %s" % (len(self.log_probs), len(self.rewards)))

        rewards = self.discount_rewards(self.rewards, self.gamma)
        rewards = config.make_variable(rewards)
        # rewards = self.normalize_rewards(rewards)  # For now nothing to normalize, standard deviation = 0

        policy_losses = [(-log_prob * reward / len(self.log_probs)) for log_prob, reward in zip(self.log_probs, rewards)]

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_losses).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.log_probs[:]

        return abs(int(policy_loss))


DEFAULT_WEIGHT_DECAY = 0.01


class FCReinforcePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None, weight_decay=DEFAULT_WEIGHT_DECAY):
        super(FCReinforcePlayer, self).__init__(strategy=strategy if strategy is not None
            else PGStrategy(lr, weight_decay=weight_decay, model=FCPolicyModel(config=config)))


class LargeFCReinforcePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None, weight_decay=DEFAULT_WEIGHT_DECAY*2):
        super(LargeFCReinforcePlayer, self).__init__(strategy=strategy if strategy is not None
            else PGStrategy(lr, weight_decay=weight_decay, model=LargeFCPolicyModel(config=config)))


class HugeFCReinforcePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None, weight_decay=DEFAULT_WEIGHT_DECAY*2):
        super(HugeFCReinforcePlayer, self).__init__(strategy=strategy if strategy is not None
            else PGStrategy(lr, weight_decay=weight_decay, model=HugeFCPolicyModel(config=config)))


class ConvReinforcePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None, weight_decay=DEFAULT_WEIGHT_DECAY):
        super(ConvReinforcePlayer, self).__init__(strategy=strategy if strategy is not None
            else PGStrategy(lr, weight_decay=weight_decay, model=ConvPolicyModel(config=config)))
