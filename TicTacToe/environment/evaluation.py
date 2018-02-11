import numpy as np

import TicTacToe.config as config
from abstractClasses import LearningPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.players.base_players import RandomPlayer, NovicePlayer, ExperiencedPlayer


def evaluate_against_base_players(player, evaluation_players=None):
    """ Standardized evaluation against base players. @return an evaluation score (0, 100) """

    EVALUATION_PLAYERS = evaluation_players if evaluation_players is not None else (RandomPlayer(), NovicePlayer(), ExperiencedPlayer())

    # Store original training values
    if issubclass(player.__class__, LearningPlayer):
        training_values = player.strategy.train, player.strategy.model.training
        player.strategy.train, player.strategy.model.training = False, False

    results = []
    accumulated_rewards = []
    for e_player in EVALUATION_PLAYERS:
        simulation = TicTacToe([player, e_player])
        reward, losses = simulation.run_simulations(config.EVALUATION_GAMES)
        results.append((e_player, reward))
        accumulated_rewards.append(reward)

    # Restore original training values
    if issubclass(player.__class__, LearningPlayer):
        player.strategy.train, player.strategy.model.training = training_values

    return np.mean(accumulated_rewards), results
