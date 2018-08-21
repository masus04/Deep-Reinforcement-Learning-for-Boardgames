import torch
from torch.nn import functional as F

import abstractClasses as abstract


class FCPolicyModel(abstract.Model):

    def __init__(self, config):
        super(FCPolicyModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        self.intermediate_size = 128

        self.fc1 = torch.nn.Linear(in_features=self.board_size**2, out_features=self.intermediate_size)
        self.fc2 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)

        self.policy_head = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.board_size ** 2)
        self.vf_head = torch.nn.Linear(in_features=self.intermediate_size, out_features=1)

        self.__xavier_initialization__()

        if config.CUDA:
            self.cuda(0)

    def forward(self, input, legal_moves_map):
        x = input.view(-1, self.board_size**2)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        p = self.policy_head(x)
        p = self.legal_softmax(p, legal_moves_map)

        vf = self.vf_head(x)

        return p, vf


class LargeFCPolicyModel(abstract.Model):

    def __init__(self, config):
        super(LargeFCPolicyModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        self.intermediate_size = 128

        self.fc1 = torch.nn.Linear(in_features=self.board_size ** 2, out_features=self.intermediate_size)
        self.fc2 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)
        self.fc3 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)

        self.policy_head = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)
        self.policy_head2 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.board_size ** 2)

        self.vf_head = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)
        self.vf_head2 = torch.nn.Linear(in_features=self.intermediate_size, out_features=1)

        self.__xavier_initialization__()

        if config.CUDA:
            self.cuda(0)

    def forward(self, input, legal_moves_map):
        x = input.view(-1, self.board_size ** 2)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        p = self.policy_head(x)
        p = self.policy_head2(p)
        p = self.legal_softmax(p, legal_moves_map)

        vf = self.vf_head(x)
        vf = self.vf_head2(vf)

        return p, vf


class HugeFCPolicyModel(abstract.Model):

    def __init__(self, config):
        super(HugeFCPolicyModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        self.intermediate_size = 128

        self.fc1 = torch.nn.Linear(in_features=self.board_size ** 2, out_features=self.intermediate_size)
        self.fc2 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)
        self.fc3 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)
        self.fc4 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)

        self.fc5 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)
        self.fc6 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)

        self.policy_head1 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)
        self.policy_head2 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.board_size ** 2)

        self.vf_head1 = torch.nn.Linear(in_features=self.intermediate_size, out_features=self.intermediate_size)
        self.vf_head2 = torch.nn.Linear(in_features=self.intermediate_size, out_features=1)

        self.__xavier_initialization__()

        if config.CUDA:
            self.cuda(0)

    def forward(self, input, legal_moves_map):
        x = input.view(-1, self.board_size ** 2)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))

        x = F.leaky_relu(self.fc5(x))
        x = F.leaky_relu(self.fc6(x))

        p = F.leaky_relu(self.policy_head1(x))
        p = self.policy_head2(p)
        p = self.legal_softmax(p, legal_moves_map)

        vf = F.leaky_relu(self.vf_head1(x))
        vf = self.vf_head2(vf)

        return p, vf


class ConvPolicyModel(abstract.Model):

    def __init__(self, config):
        super(ConvPolicyModel, self).__init__()

        self.board_size = config.BOARD_SIZE
        self.self.conv_channels = 32

        # Create representation
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)
        self.conv8 = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=self.conv_channels, kernel_size=3, padding=1)

        # Evaluate and output move possibilities
        self.reduce = torch.nn.Conv2d(in_channels=self.conv_channels, out_channels=1, kernel_size=1, padding=0)

        self.vf_head = torch.nn.Linear(in_features=self.board_size**2, out_features=1)

        self.__xavier_initialization__()

    def forward(self, input, legal_moves_map):
        x = input.unsqueeze(dim=0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        # TODO: Give more flexibility to both outputs!?
        x = self.reduce(x)
        x = x.view(-1, self.board_size**2)

        p = self.legal_softmax(x, legal_moves_map)
        vf = self.vf_head(x)

        return p, vf
