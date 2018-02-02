import numpy as np

import TicTacToe.config as config
from abstractClasses import LearningPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.players.base_players import RandomPlayer, ExperiencedPlayer


def evaluate_against_base_players(player):
    """ Standardized evaluation against base players. @return an evaluation score (0, 100) """

    EVALUATION_PLAYERS = (RandomPlayer(), ExperiencedPlayer())

    # Store original training values
    if issubclass(player.__class__, LearningPlayer):
        training_values = player.strategy.train, player.strategy.model.training
        player.strategy.train, player.strategy.model.training = False, False

    accumulated_rewards = []
    for e_player in EVALUATION_PLAYERS:
        simulation = TicTacToe([player, e_player])
        reward, losses = simulation.run_simulations(config.EVALUATION_GAMES)
        accumulated_rewards.append(reward)

    # Restore original training values
    if issubclass(player.__class__, LearningPlayer):
        player.strategy.train, player.strategy.model.training = training_values

    return np.mean(accumulated_rewards)
