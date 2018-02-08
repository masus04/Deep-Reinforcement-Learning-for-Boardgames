import random
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import TicTacToe.config as config
import abstractClasses as abstract
from abstractClasses import PlayerException


class ReinforcePlayer(abstract.LearningPlayer):

    def __init__(self, strategy, lr, batch_size=1):
        super(ReinforcePlayer, self).__init__()

        if issubclass(strategy, abstract.Strategy):
            self.strategy = strategy(lr=lr, batch_size=batch_size)
        elif issubclass(strategy.__class__, abstract.Strategy):
            self.strategy = strategy
        else:
            raise Exception("ReinforcePlayer takes as a strategy argument a subclass of %s, received %s" % (abstract.Model, strategy))

    def get_move(self, board):
        if self.strategy.train:
            self.num_moves += 1
        try:
            return self.strategy.evaluate(board.get_representation(self.color), board.get_legal_moves_map(self.color))
        except PlayerException as e:
            print(e)
            random.choice(board.get_valid_moves())

    def register_winner(self, winner_color):
        self.strategy.rewards += ([self.get_label(winner_color)] * self.num_moves)
        self.num_moves = 0
        return self.strategy.update()


class PGStrategy(abstract.Strategy):

    def __init__(self, lr, batch_size, gamma=config.GAMMA,  model=None):
        super(PGStrategy, self).__init__()
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.model = model if model else PGLargeFCModel()  # PGFCModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.batches = []

    def evaluate(self, board_sample, legal_moves_map):
        input = config.make_variable(torch.FloatTensor([board_sample]))
        legal_moves_map = config.make_variable(legal_moves_map)
        probs = self.model(input, legal_moves_map)

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
        self.batches.append(policy_loss)

        if len(self.batches) >= self.batch_size:
            batch_loss = torch.cat(self.batches).sum()/len(self.batches)
            batch_loss.backward()
            self.optimizer.step()
            del self.batches[:]

        del self.rewards[:]
        del self.log_probs[:]

        return abs(policy_loss.data[0])

    @staticmethod
    def discount_rewards(rewards, discount_factor):
        if discount_factor <= 0:
            return deepcopy(rewards)

        running_reward = 0
        discounted_rewards = []
        for r in rewards[::-1]:
            running_reward = config.GAMMA * running_reward + r
            discounted_rewards.insert(0, running_reward)

        return discounted_rewards

    @staticmethod
    def normalize_rewards(rewards):
        return (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps)


class PGFCModel(abstract.Model):

    def __init__(self):
        super(PGFCModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        intermediate_size = 128
        self.p_dropout = 0.0

        self.fc1 = torch.nn.Linear(in_features=self.board_size**2, out_features=intermediate_size)
        self.dropout1 = torch.nn.Dropout(p=self.p_dropout)
        self.fc2 = torch.nn.Linear(in_features=intermediate_size, out_features=intermediate_size)
        self.dropout2 = torch.nn.Dropout(p=self.p_dropout)
        self.fc3 = torch.nn.Linear(in_features=intermediate_size, out_features=self.board_size ** 2)

        self.__xavier_initialization__()

        if config.CUDA:
            self.cuda(0)

    def forward(self, input, legal_moves_map):
        x = input.view(-1, self.board_size**2)
        x = F.relu(self.fc1(x))
        if self.p_dropout > 0:
            x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        if self.p_dropout > 0:
            x = self.dropout2(x)
        x = self.fc3(x)

        x = self.legal_softmax(x, legal_moves_map)
        return x


class PGLargeFCModel(abstract.Model):
    def __init__(self):
        super(PGLargeFCModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        intermediate_size = 128
        self.p_dropout = 0.5

        self.fc1 = torch.nn.Linear(in_features=self.board_size ** 2, out_features=intermediate_size)
        self.dropout1 = torch.nn.Dropout(p=self.p_dropout)
        self.fc2 = torch.nn.Linear(in_features=intermediate_size, out_features=intermediate_size)
        self.dropout2 = torch.nn.Dropout(p=self.p_dropout)
        self.fc3 = torch.nn.Linear(in_features=intermediate_size, out_features=intermediate_size)
        self.dropout3 = torch.nn.Dropout(p=self.p_dropout)
        self.fc4 = torch.nn.Linear(in_features=intermediate_size, out_features=intermediate_size)
        self.dropout4 = torch.nn.Dropout(p=self.p_dropout)
        self.fc5 = torch.nn.Linear(in_features=intermediate_size, out_features=self.board_size ** 2)

        self.__xavier_initialization__()

        if config.CUDA:
            self.cuda(0)

    def forward(self, input, legal_moves_map):
        x = input.view(-1, self.board_size ** 2)
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        # x = self.dropout3(x)
        # x = F.relu(self.fc4(x))
        # x = self.dropout4(x)
        x = self.fc5(x)

        x = self.legal_softmax(x, legal_moves_map)
        return x


class PGConvModel(abstract.Model):

    def __init__(self):
        super(PGConvModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        self.conv_channels = 8

        # Create representation
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)

        # Evaluate and output move possibilities
        self.reduce = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=1, kernel_size=1, padding=0)

        self.__xavier_initialization__()

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.reduce(x)
        x = x.view(-1, self.board_size**2)
        return x
