import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from numba import jit

import TicTacToe.config as config
from models8x8 import FCPolicyModel, LargeFCPolicyModel, ConvPolicyModel
from abstractClasses import LearningPlayer, Strategy, PlayerException


class ACStrategy(Strategy):

    def __init__(self, lr, gamma=config.GAMMA, model=None):
        super(ACStrategy, self).__init__()
        self.lr = lr
        self.gamma = gamma

        self.model = model if model else FCPolicyModel(ac_policy=True)
        self.model.ac_policy = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.state_values = []
        self.board_samples = []
        self.legal_moves = []

    def evaluate(self, board_sample, legal_moves_map):
        input = config.make_variable(torch.FloatTensor([board_sample]))

        probs, state_value = self.model(input, config.make_variable(legal_moves_map))
        distribution = Categorical(probs)
        action = distribution.sample()

        move = (action.data[0] // config.BOARD_SIZE, action.data[0] % config.BOARD_SIZE)
        if self.train:
            self.log_probs.append(distribution.log_prob(action))
            self.state_values.append(state_value)
            self.board_samples.append(board_sample)
            self.legal_moves.append(legal_moves_map)
        return move

    def update(self):
        # ---------------------- Error Logging ---------------------- #
        if not self.train:
            return 0

        if len(self.log_probs) != len(self.rewards) or len(self.log_probs) != len(self.state_values):
            raise PlayerException("log_probs length must be equal to rewards length as well as state_values length. Got %s - %s - %s" % (len(self.log_probs), len(self.rewards), len(self.state_values)))

        # ----------------------------------------------------------- #
        # Use either Discount (and baseline),
        rewards = self.discount_rewards(self.rewards, self.gamma)
        rewards = self.rewards_baseline(rewards)
        # Or Bootstrapping
        # rewards = self.bootstrap_rewards()
        rewards = config.make_variable(torch.FloatTensor(rewards))
        # rewards = self.normalize_rewards(rewards)  # For now nothing to normalize, standard deviation = 0

        loss = calculate_loss(self.log_probs, self.state_values, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.log_probs[:]
        del self.state_values[:]
        del self.board_samples[:]

        return abs(loss.data[0])


class FCBaselinePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None):
        super(FCBaselinePlayer, self).__init__(strategy=strategy if strategy is not None
                                         else ACStrategy(lr, model=FCPolicyModel(ac_policy=True)))


class LargeFCBaselinePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None):
        super(LargeFCBaselinePlayer, self).__init__(strategy=strategy if strategy is not None
                                         else ACStrategy(lr, model=LargeFCPolicyModel(ac_policy=True)))


class ConvBaselinePlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None):
        super(ConvBaselinePlayer, self).__init__(strategy=strategy if strategy is not None
                                           else ACStrategy(lr, model=ConvPolicyModel(ac_policy=True)))


@jit
def calculate_loss(log_probs, state_values, rewards):
    policy_losses = []
    value_losses = []

    for log_prob, state_value, reward in zip(log_probs, state_values, rewards):
        policy_losses.append(-log_prob * (reward - state_value.data[0][0]))
        value_losses.append(F.smooth_l1_loss(state_value, reward))

    return torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
