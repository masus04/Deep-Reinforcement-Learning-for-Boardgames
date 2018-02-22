import torch
from torch.distributions import Categorical
from copy import deepcopy

import TicTacToe.config as config
from abstractClasses import LearningPlayer, Strategy, PlayerException
from TicTacToe.players.models import FCPolicyModel, LargeFCPolicyModel, LargeValueFunctionModel, ConvPolicyModel, ConvValueFunctionModel


class ActorCriticPlayer(LearningPlayer):
    def __init__(self, strategy, lr, batch_size=1):
        super(ActorCriticPlayer, self).__init__()

        if issubclass(strategy, Strategy):
            self.strategy = strategy(lr=lr, batch_size=batch_size)
        elif issubclass(strategy.__class__, Strategy):
            self.strategy = strategy
        else:
            raise Exception("ReinforcePlayer takes as a strategy argument a subclass of %s, received %s" % (Model, strategy))

    def get_move(self, board):
        representation = board.get_representation(self.color)
        legal_moves_map = board.get_legal_moves_map(self.color)
        if self.strategy.train:
            self.strategy.rewards.append(0)
            self.strategy.board_samples.append(representation)
        return self.strategy.evaluate(representation, legal_moves_map)

    def register_winner(self, winner_color):
        if self.strategy.train:
            self.strategy.rewards[-1] = self.get_label(winner_color)

        return self.strategy.update()


class ACStrategy(Strategy):
    def __init__(self, lr, batch_size, gamma=config.GAMMA, alpha=config.ALPHA, policy=None, value_function=None):
        super(ACStrategy, self).__init__()
        self.lr = lr  # learning rate
        self.gamma = gamma  # reward discounting factor
        self.alpha = alpha  # bootstrapping factor
        self.batch_size = batch_size
        self.policy = policy if policy else FCPolicyModel()
        self.value_function = value_function if value_function else FCPolicyModel()
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.vf_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.vf_criterion = torch.nn.MSELoss()

        self.board_samples = []

        self.policy_batch_losses = []
        self.vf_batch_losses = []

    def evaluate(self, board_sample, legal_moves_map):
        input = config.make_variable(torch.FloatTensor([board_sample]))
        legal_moves_map = config.make_variable(legal_moves_map)
        probs = self.policy(input, legal_moves_map)

        try:
            distribution = Categorical(probs)
            action = distribution.sample()
        except RuntimeError:
            self.policy(input, legal_moves_map)
            raise PlayerException("sum(probs) <= 0:\n%s\n board:\n%s\nlegal moves:\n%s" % (probs, board_sample, legal_moves_map))

        move = (action.data[0] // config.BOARD_SIZE, action.data[0] % config.BOARD_SIZE)
        log_prob = distribution.log_prob(action)
        if self.train:
            self.log_probs.append(log_prob)
        return move

    def update(self):
        if not self.train:
            return 0

        if len(self.log_probs) != len(self.rewards) or len(self.log_probs) != len(self.board_samples):
            raise PlayerException("log_probs, rewards and board_samples must all have the same length. Got %s - %s - %s" % (len(self.log_probs), len(self.rewards), len(self.board_samples)))

        rewards = self.discount_rewards(self.rewards, self.gamma)
        rewards = self.bootstrap_rewards(rewards)
        rewards = config.make_variable(rewards)
        # rewards = self.normalize_rewards(rewards)  # For now nothing to normalize, standard deviation = 0

        """ Policy update """
        policy_losses = [(-log_prob * reward) for log_prob, reward in zip(self.log_probs, rewards)]
        policy_loss = torch.cat(policy_losses).sum() / len(policy_losses)
        self.policy_batch_losses.append(policy_loss)

        if len(self.policy_batch_losses) >= self.batch_size:
            self.policy_optimizer.zero_grad()
            batch_loss = torch.cat(self.policy_batch_losses).sum() / len(self.policy_batch_losses)
            batch_loss.backward()
            self.policy_optimizer.step()
            del self.policy_batch_losses[:]

        """ Value function update """

        # TODO: Verify structure
        vf_losses = [self.vf_criterion.forward(self.value_function(config.make_variable(sample)), reward) for sample, reward in zip(self.board_samples, rewards)]
        vf_loss = torch.cat(vf_losses).sum() / len(vf_losses)
        self.vf_batch_losses.append(vf_loss)

        if len(self.vf_batch_losses) >= self.batch_size:
            self.vf_optimizer.zero_grad()
            batch_loss = torch.cat(self.vf_batch_losses).sum() / len(self.vf_batch_losses)
            batch_loss.backward()
            self.vf_optimizer.step()
            del self.vf_batch_losses[:]

        del self.rewards[:]
        del self.log_probs[:]
        del self.board_samples[:]

        return abs(policy_loss.data[0])

    def bootstrap_rewards(self, rewards):
        rewards = [self.rewards[i] +
                   self.alpha * self.value_function(config.make_variable(self.board_samples[i+1])).data[0][0] -
                   self.value_function(config.make_variable(self.board_samples[i])).data[0][0]
                   for i in range(len(rewards)-1)]
        rewards.append(self.rewards[-1])
        return rewards

    # lr, batch_size, gamma=config.GAMMA, alpha=config.ALPHA, policy=None, value_function=None)
    def copy(self, shared_weights=True):
        if shared_weights:
            strategy = self.__class__(lr=self.lr, batch_size=self.batch_size, gamma=self.gamma, alpha=self.alpha, policy=self.policy, value_function=self.value_function)
        else:
            strategy = self.__class__(lr=self.lr, batch_size=self.batch_size, gamma=self.gamma, alpha=self.alpha, policy=self.policy.copy(shared_weights), value_function=self.value_function.copy(shared_weights))

        strategy.train = deepcopy(self.train)
        strategy.log_probs = deepcopy(self.log_probs)

        strategy.policy_batch_losses = deepcopy(self.policy_batch_losses)
        strategy.vf_batch_losses = deepcopy(self.vf_batch_losses)
        return strategy


class FCActorCriticPlayer(ActorCriticPlayer):
    def __init__(self, lr, strategy=None, batch_size=1):
        super(FCActorCriticPlayer, self).__init__(lr=lr, strategy=strategy if strategy is not None
                                                  else ACStrategy(lr, batch_size, policy=LargeFCPolicyModel(), value_function=LargeValueFunctionModel()))


class ConvActorCriticPlayer(ActorCriticPlayer):
    def __init__(self, lr, strategy=None, batch_size=1):
        super(FCActorCriticPlayer, self).__init__(lr=lr, strategy=strategy if strategy is not None
                                                  else ACStrategy(lr, batch_size, policy=ConvPolicyModel(), value_function=ConvValueFunctionModel()))
