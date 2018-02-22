import torch
from torch.nn import functional as F

import abstractClasses as abstract
from TicTacToe import config as config


class FCPolicyModel(abstract.Model):

    def __init__(self):
        super(FCPolicyModel, self).__init__()

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


class LargeFCPolicyModel(abstract.Model):
    def __init__(self):
        super(LargeFCPolicyModel, self).__init__()

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


class ConvPolicyModel(abstract.Model):

    def __init__(self):
        super(ConvPolicyModel, self).__init__()

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

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))

        x = self.reduce(x)
        x = x.view(-1, self.board_size**2)

        x = self.legal_softmax(x, legal_moves_map)

        return x


class LargeValueFunctionModel(LargeFCPolicyModel):
    def __init__(self):
        super(LargeValueFunctionModel, self).__init__()
        self.vf_output = torch.nn.Linear(in_features=self.board_size ** 2, out_features=1)

    def forward(self, input):
        x = input.view(-1, self.board_size ** 2)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))

        x = self.vf_output(x)
        return x


class ConvValueFunctionModel(ConvPolicyModel):
    def __init__(self):
        super(ConvValueFunctionModel, self).__init__()
        self.vf_output = torch.nn.Linear(in_features=self.board_size ** 2, out_features=1)

    def forward(self, input, legal_moves_map):
        x = input.unsqueeze(dim=0)

        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.reduce(x)
        x = x.view(-1, self.board_size ** 2)

        x = self.vf_output(x)
        return x
