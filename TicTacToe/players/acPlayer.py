import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from numba import jit

import TicTacToe.config as config
from models import FCPolicyModel, LargeFCPolicyModel, ConvPolicyModel
from abstractClasses import LearningPlayer, Strategy, PlayerException


class ACStrategy(Strategy):

    def __init__(self, lr, weight_decay, gamma=config.GAMMA, model=None):
        super(ACStrategy, self).__init__()
        self.lr = lr
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.online = False

        self.model = model if model else FCPolicyModel(config=config)
        self.model.double()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.state_values = []
        self.board_samples = []
        self.legal_moves = []

    def evaluate(self, board_sample, legal_moves_map):
        input = config.make_variable([board_sample])

        probs, state_value = self.model(input, config.make_variable(legal_moves_map))
        distribution = Categorical(probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        move = (action // config.BOARD_SIZE, action % config.BOARD_SIZE)
        if self.train:
            if self.online and self.state_values:
                self.online_policy_update(board_sample, legal_moves_map, log_prob)

            self.log_probs.append(log_prob)
            self.state_values.append(state_value[0])
            self.board_samples.append(board_sample)
            self.legal_moves.append(legal_moves_map)
        return move

    def update(self):
        # ---------------------- Error Logging ---------------------- #
        if not self.train:
            return None

        if len(self.log_probs) != len(self.rewards) or len(self.log_probs) != len(self.state_values):
            raise PlayerException("log_probs length must be equal to rewards length as well as state_values length. Got %s - %s - %s" % (len(self.log_probs), len(self.rewards), len(self.state_values)))
        # ----------------------------------------------------------- #

        # Bootstrapping
        rewards = self.bootstrap_rewards()
        rewards = config.make_variable(rewards)
        # rewards = self.normalize_rewards(rewards)  # For now nothing to normalize, standard deviation = 0

        if self.online:
            loss = calculate_online_loss(self.state_values, rewards)
        else:
            loss = calculate_loss(self.log_probs, self.state_values, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del self.rewards[:]
        del self.log_probs[:]
        del self.state_values[:]
        del self.board_samples[:]
        del self.legal_moves[:]

        return abs(loss.data)

    def online_policy_update(self, board, legal_moves, logprob):
        new_value = self.model(config.make_variable([board]), config.make_variable([legal_moves]))[1].data[0,0]
        reward = self.state_values[-1].data[0, 0] - new_value
        loss = -logprob * reward

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()


class FCACPlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None, weight_decay=0.003):
        super(FCACPlayer, self).__init__(strategy=strategy if strategy is not None
                                         else ACStrategy(lr, weight_decay=weight_decay, model=FCPolicyModel(config=config)))


class LargeFCACPlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None, weight_decay=0.005):
        super(LargeFCACPlayer, self).__init__(strategy=strategy if strategy is not None
                                         else ACStrategy(lr, weight_decay=weight_decay, model=LargeFCPolicyModel(config=config)))


class ConvACPlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None, weight_decay=0.003):
        super(ConvACPlayer, self).__init__(strategy=strategy if strategy is not None
                                           else ACStrategy(lr, weight_decay=weight_decay, model=ConvPolicyModel(config=config)))


@jit
def calculate_loss(log_probs, state_values, rewards):
    policy_losses = []
    value_losses = []

    for log_prob, state_value, reward in zip(log_probs, state_values, rewards):
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(state_value, reward))

    return torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()


@jit
def calculate_online_loss(state_values, rewards):
    value_losses = []

    for state_value, reward in zip(state_values, rewards):
        value_losses.append(F.smooth_l1_loss(state_value, reward))

    return torch.stack(value_losses).sum()
