import numpy as np

import TicTacToe.config as config
from abstractClasses import LearningPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.players.base_players import RandomPlayer, NovicePlayer, ExperiencedPlayer


def evaluate_against_base_players(player, evaluation_players=[RandomPlayer(), NovicePlayer(), ExperiencedPlayer()]):
    """
    Standardized evaluation against base players.


    :param player: The player to be evaluated
    :param evaluation_players: A list of players against which the player should be evaluated
    :return: a tuple (score, results) where score is the average score over all evaluation games (scalar (-1, 1)) and results is a list of
             tuples (name, score) where 'score' is the score (-1, 1) achieved when evaluated against the player named 'name'.
    """

    evaluation_players

    # Store original training values
    if issubclass(player.__class__, LearningPlayer):
        training_values = player.strategy.train, player.strategy.model.training
        player.strategy.train, player.strategy.model.training = False, False

    results = []
    for e_player in evaluation_players:
        simulation = TicTacToe([player, e_player])
        reward, losses = simulation.run_simulations(config.EVALUATION_GAMES)
        results.append((e_player.__class__.__name__, reward))

    results = [(result[0], np.mean(result[1])) for result in results]
    results.insert(0, ("Total Score", np.mean([res[1] for res in results])))  # Insert average overall score as first element of re§sults

    # Restore original training values
    if issubclass(player.__class__, LearningPlayer):
        player.strategy.train, player.strategy.model.training = training_values

    return results[0][1], results
