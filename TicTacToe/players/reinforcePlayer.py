
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
        self.model = model if model else PGModel()
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


class PGModel(abstract.Model):

    def __init__(self):
        super(PGModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        game_states = self.board_size**2**3
        intermediate_size = 128

        self.fc1 = torch.nn.Linear(in_features=self.board_size**2, out_features=intermediate_size)
        # self.fc2 = torch.nn.Linear(in_features=intermediate_size, out_features=self.board_size**2)
        self.fc2 = torch.nn.Linear(in_features=intermediate_size, out_features=intermediate_size)
        self.fc3 = torch.nn.Linear(in_features=intermediate_size, out_features=self.board_size ** 2)

        self.__xavier_initialization__()

    def forward(self, input):
        x = input.view(-1, self.board_size**2)
        x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = F.softmax(x, dim=1)
        return x
