import torch
import torch.nn.functional as F
from torch.autograd.variable import torch
from torch.distributions import Categorical
from torch.autograd import Variable
from copy import deepcopy

import TicTacToe.config as config
import abstractClasses as abstract


class ReinforcePlayer(abstract.LearningPlayer):

    def __init__(self, strategy, lr):
        super(ReinforcePlayer, self).__init__()

        if issubclass(strategy, abstract.Strategy):
            self.strategy = strategy(lr=lr)
        elif issubclass(strategy.__class__, abstract.Strategy):
            self.strategy = strategy
        else:
            raise Exception("ReinforcePlayer takes as a strategy argument a subclass of %s, received %s" % (abstract.Model, strategy))

    def get_move(self, board):
        return self.strategy.evaluate(board.get_representation(self.color))

    def register_winner(self, winner_color):
        return self.strategy.update(self.get_label(winner_color))


class PGStrategy(abstract.Strategy):

    def __init__(self, lr, model=None):
        super(PGStrategy, self).__init__()
        self.lr = lr
        self.model = model if model else PGLinearModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def evaluate(self, board_sample):
        input = Variable(torch.FloatTensor([board_sample]))
        probs = self.model(input)

        distribution = Categorical(probs)
        action = distribution.sample()

        move = (action.data[0] // config.BOARD_SIZE, action.data[0] % config.BOARD_SIZE)
        log_prob = distribution.log_prob(action)
        if self.train:
            self.training_samples.append(log_prob)
        return move

    def update(self, training_label):
        if self.train:

            self.optimizer.zero_grad()
            label = Variable(torch.FloatTensor([training_label]))

            loss = -torch.cat(self.training_samples).sum()
            policy_loss = loss * label
            policy_loss.backward()

            self.training_samples = []
            self.optimizer.step()

            return loss.data[0]


class PGLinearModel(abstract.Model):

    def __init__(self):
        super(PGLinearModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        intermediate_size = 128
        self.p_dropout = 0.0

        self.fc1 = torch.nn.Linear(in_features=self.board_size**2, out_features=intermediate_size)
        self.dropout1 = torch.nn.Dropout(p=self.p_dropout)
        self.fc3 = torch.nn.Linear(in_features=intermediate_size, out_features=self.board_size ** 2)

        self.__xavier_initialization__()

    def forward(self, input):
        x = input.view(-1, self.board_size**2)
        x = F.relu(self.fc1(x))
        if self.p_dropout > 0:
            x = self.dropout1(x)
        x = self.fc3(x)

        x = F.softmax(x, dim=1)
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
        x = F.softmax(x, dim=1)
        return x
