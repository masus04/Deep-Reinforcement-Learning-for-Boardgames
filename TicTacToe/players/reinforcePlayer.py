
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

import TicTacToe.config as config
from TicTacToe.environment.board import TicTacToeBoard
import abstract_classes as abstract


class ReinforcePlayer(abstract.Player):

    def __init__(self, strategy, lr):
        if not issubclass(strategy, abstract.Strategy):
            raise Exception("ReinforcePlayer takes as a strategy argument a subclass of %s, received %s" % (abstract.Model, strategy))
        self.strategy = strategy(lr=lr)

    def get_move(self, board):
        return self.strategy.evaluate(board.get_representation(self.color))

    def register_winner(self, winner_color):
        self.strategy.update(self.get_label(winner_color))


class Strategy(abstract.Strategy):

    def __init__(self, lr):
        self.lr = lr
        self.model = Model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def evaluate(self, board_sample):
        input = Variable(torch.FloatTensor([board_sample]))
        probs = self.model(input)

        distribution = Categorical(probs)
        action = distribution.sample()

        move = (action.data[0] // config.BOARD_SIZE, action.data[0] % config.BOARD_SIZE)
        log_prob = distribution.log_prob(action)
        if self.model.training:
            self.training_samples.append(log_prob)
        return move

    def update(self, training_label):
        if self.model.training:

            self.optimizer.zero_grad()
            label = Variable(torch.FloatTensor([training_label]))

            policy_loss = -torch.cat(self.training_samples).sum() * label
            policy_loss.backward()

            self.training_samples = []
            self.optimizer.step()


class Model(abstract.Model):

    def __init__(self):
        super(Model, self).__init__()

        self.board_size = config.BOARD_SIZE
        board_states = self.board_size**2**3

        self.fc1 = torch.nn.Linear(in_features=self.board_size**2, out_features=board_states)
        self.fc2 = torch.nn.Linear(in_features=board_states, out_features=self.board_size**2)

    def forward(self, input):
        x = input.view(-1, self.board_size**2)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.softmax(x, dim=1)
        return x
