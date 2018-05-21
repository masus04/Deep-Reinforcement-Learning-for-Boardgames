import torch
from torch.distributions import Categorical

import Othello.config as config
import abstractClasses as abstract
from abstractClasses import LearningPlayer
from models8x8 import FCPolicyModel, LargeFCPolicyModel, HugeFCPolicyModel, ConvPolicyModel
from abstractClasses import PlayerException


class PGStrategy(abstract.Strategy):

    def __init__(self, lr, gamma=config.GAMMA,  model=None):
        super(PGStrategy, self).__init__()
        self.lr = lr
        self.gamma = gamma

        self.model = model if model else FCPolicyModel()  # PGFCModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


    def evaluate(self, board_sample, legal_moves_map):
        input = config.make_variable(torch.FloatTensor([board_sample]))
        legal_moves_map = config.make_variable(legal_moves_map)
        probs, _ = self.model(input, legal_moves_map)

        try:
            distribution = Categorical(probs)
            action = distribution.sample()
        except RuntimeError:
            self.model(input, legal_moves_map)
            raise PlayerException("sum(probs) <= 0:\n%s\n board:\n%s\nlegal moves:\n%s" % (probs, board_sample, legal_moves_map))

        move = (action.data[0] // config.BOARD_SIZE, action.data[0] % config.BOARD_SIZE)
        log_prob = distribution.log_prob(action)
        if self.train:
            self.log_probs.append(log_prob)
        return move

    def update(self):
        if not self.train:
            return 0

        if len(self.log_probs) != len(self.rewards):
            raise abstract.PlayerException("log_probs length must be equal to rewards length. Got %s - %s" % (len(self.log_probs), len(self.rewards)))

        rewards = self.discount_rewards(self.rewards, self.gamma)
        rewards = config.make_variable(torch.FloatTensor(rewards))
        # rewards = self.normalize_rewards(rewards)  # For now nothing to normalize, standard deviation = 0

        policy_losses = [(-log_prob * reward) for log_prob, reward in zip(self.log_probs, rewards)]

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_losses).sum()/len(policy_losses)
        policy_loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.log_probs[:]

        return abs(policy_loss.data[0])


class FCReinforcePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None):
        super(FCReinforcePlayer, self).__init__(strategy=strategy if strategy is not None
                                                else PGStrategy(lr, model=FCPolicyModel()))


class LargeFCReinforcePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None):
        super(LargeFCReinforcePlayer, self).__init__(strategy=strategy if strategy is not None
                                                else PGStrategy(lr, model=LargeFCPolicyModel()))


class HugeFCReinforcePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None):
        super(HugeFCReinforcePlayer, self).__init__(strategy=strategy if strategy is not None
                                                else PGStrategy(lr, model=HugeFCPolicyModel()))


class ConvReinforcePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None):
        super(ConvReinforcePlayer, self).__init__(strategy=strategy if strategy is not None
                                                  else PGStrategy(lr, model=ConvPolicyModel()))
