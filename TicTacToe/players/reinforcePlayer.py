import random
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import TicTacToe.config as config
import abstractClasses as abstract
from abstractClasses import PlayerException
from modules import swish


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
            self.strategy.rewards.append(0)
        return self.strategy.evaluate(board.get_representation(self.color), board.get_legal_moves_map(self.color))

    def register_winner(self, winner_color):
        if self.strategy.train:
            self.strategy.rewards[-1] = self.get_label(winner_color)

        return self.strategy.update()


class PGStrategy(abstract.Strategy):

    def __init__(self, lr, batch_size, gamma=config.GAMMA,  model=None):
        super(PGStrategy, self).__init__()
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.model = model if model else PGFCModel()  # PGFCModel()
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
    def normalize_rewards(rewards):
        return (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps)


class PGFCModel(abstract.Model):

    def __init__(self):
        super(PGFCModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        intermediate_size = 128

        self.fc1 = torch.nn.Linear(in_features=self.board_size**2, out_features=intermediate_size)
        self.fc2 = torch.nn.Linear(in_features=intermediate_size, out_features=intermediate_size)
        self.fc3 = torch.nn.Linear(in_features=intermediate_size, out_features=self.board_size ** 2)

        self.__xavier_initialization__()

        if config.CUDA:
            self.cuda(0)

    def forward(self, input, legal_moves_map):
        x = input.view(-1, self.board_size**2)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        x = self.legal_softmax(x, legal_moves_map)
        return x


class PGLargeFCModel(abstract.Model):
    def __init__(self):
        super(PGLargeFCModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        intermediate_size = 128

        self.fc1 = torch.nn.Linear(in_features=self.board_size ** 2, out_features=intermediate_size)
        self.fc2 = torch.nn.Linear(in_features=intermediate_size, out_features=intermediate_size)
        self.fc3 = torch.nn.Linear(in_features=intermediate_size, out_features=intermediate_size)
        self.fc4 = torch.nn.Linear(in_features=intermediate_size, out_features=intermediate_size)
        self.fc5 = torch.nn.Linear(in_features=intermediate_size, out_features=self.board_size ** 2)

        self.__xavier_initialization__()

        if config.CUDA:
            self.cuda(0)

    def forward(self, input, legal_moves_map):
        x = input.view(-1, self.board_size ** 2)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
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

    def forward(self, input, legal_moves_map):
        x = input.unsqueeze(dim=0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.reduce(x)
        x = x.view(-1, self.board_size**2)

        x = self.legal_softmax(x, legal_moves_map)

        return x


class FCReinforcePlayer(ReinforcePlayer):
    def __init__(self, lr, model=PGLargeFCModel(), batch_size=1):
        super(FCReinforcePlayer, self).__init__(strategy=PGStrategy(lr, batch_size, model=model), lr=lr)


class ConvReinforcePlayer(ReinforcePlayer):
    def __init__(self, lr, model=PGConvModel(), batch_size=1):
        super(ConvReinforcePlayer, self).__init__(strategy=PGStrategy(lr, batch_size, model=model), lr=lr)
