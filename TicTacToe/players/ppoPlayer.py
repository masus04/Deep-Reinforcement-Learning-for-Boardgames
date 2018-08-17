import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from numba import jit

import TicTacToe.config as config
from models3x3 import FCPolicyModel, LargeFCPolicyModel
from abstractClasses import LearningPlayer, Strategy, PlayerException


class PPOStrategy(Strategy):

    def __init__(self, lr, gamma=config.GAMMA, model=None):
        super(PPOStrategy, self).__init__()
        self.lr = lr
        self.gamma = gamma

        self.model = model if model else FCPolicyModel(config=config)

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
        UPDATES = 16

        # ---------------------- Error Logging ---------------------- #
        if not self.train:
            return 0

        if len(self.log_probs) != len(self.rewards) or len(self.log_probs) != len(self.state_values):
            raise PlayerException(
                "log_probs length must be equal to rewards length as well as state_values length. Got %s - %s - %s" % (len(self.log_probs), len(self.rewards), len(self.state_values)))

        # ----------------------------------------------------------- #

        old_log_probs = [log_probs.data[0] for log_probs in self.log_probs]

        rewards = self.discount_rewards(self.rewards, 0.95)  # self.gamma)
        rewards = config.make_variable(torch.FloatTensor(rewards))
        samples = list(zip(self.board_samples, self.legal_moves))
        # rewards = self.normalize_rewards(rewards)  # For now nothing to normalize, standard deviation = 0

        for i in range(UPDATES):
            loss = calculate_loss(self.log_probs, old_log_probs, self.state_values, rewards, config.CLIP)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del self.log_probs[:]
            del self.state_values[:]

            if i < UPDATES-1:  # Prepare for next iteration
                self.board_samples = []
                self.legal_moves = []
                for sample, legal_map in samples:
                    self.evaluate(sample, legal_map)

        del self.rewards[:]
        del self.board_samples[:]
        del self.legal_moves[:]

        return abs(loss.data[0])


class FCPPOPlayer(LearningPlayer):
    def __init__(self, lr=config.LR, strategy=None):
        super(FCPPOPlayer, self).__init__(strategy=strategy if strategy is not None
                                          else PPOStrategy(lr, model=LargeFCPolicyModel(config=config)))


@jit
def calculate_loss(log_probs, old_log_probs, state_values, rewards, clip):
    policy_losses = []
    value_losses = []

    for log_prob, old_log_prob, state_value, reward in zip(log_probs, old_log_probs, state_values, rewards):
        ratio = (log_prob - config.make_variable([old_log_prob])).exp()
        surr1 = ratio * (reward-state_value.data[0][0])
        surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * (reward-state_value.data[0][0])
        policy_losses.append(-torch.min(surr1, surr2))

        value_losses.append(F.smooth_l1_loss(state_value, reward))

    return torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()
