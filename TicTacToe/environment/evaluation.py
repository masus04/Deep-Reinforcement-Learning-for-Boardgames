import numpy as np
from collections import Counter

import TicTacToe.config as config
from abstractClasses import LearningPlayer
from TicTacToe.environment.game import TicTacToe
from TicTacToe.players.base_players import RandomPlayer, NovicePlayer, ExperiencedPlayer


def evaluate_against_base_players(player, evaluation_players=[RandomPlayer(), NovicePlayer(), ExperiencedPlayer()], silent=True):
    """
    Standardized evaluation against base players.


    :param player: The player to be evaluated
    :param evaluation_players: A list of players against which the player should be evaluated
    :return: a tuple (score, results) where score is the average score over all evaluation games (scalar (-1, 1)) and results is a list of
             tuples (name, score) where 'score' is the score (-1, 1) achieved when evaluated against the player named 'name'.
    """

    # Store original training values
    if issubclass(player.__class__, LearningPlayer):
        training_values = player.strategy.train, player.strategy.model.training
        player.strategy.train, player.strategy.model.training = False, False

    results = []
    for e_player in evaluation_players:
        simulation = TicTacToe([player, e_player])
        rewards, losses = simulation.run_simulations(config.EVALUATION_GAMES)
        results.append((e_player.__class__.__name__, rewards))

        if not silent:
            print_results(e_player, rewards)

    results = [(result[0], np.mean(result[1])) for result in results]
    results.insert(0, ("Total Score", np.mean([res[1] for res in results])))  # Insert average overall score as first element of results

    # Restore original training values
    if issubclass(player.__class__, LearningPlayer):
        player.strategy.train, player.strategy.model.training = training_values

    if not silent:
        print("Overall score: %s" % results[0][1])

    return results[0][1], results


def print_results(player, rewards):
    counter = Counter(rewards)
    print("Evaluating vs %s" % player.__class__.__name__)
    print("Total score: %s" % np.mean(rewards))
    print("W/D/L: %s/%s/%s\n" % (counter[config.LABEL_WIN], counter[config.LABEL_DRAW], counter[config.LABEL_LOSS]))
