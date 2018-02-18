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
    :param silent: Flag controlling if output is written to console
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
        results.append((e_player.__str__(), rewards))

        if not silent:
            print_results(player, e_player, rewards)

    results = [(result[0], np.mean(result[1])) for result in results]
    results.insert(0, ("Total Score", np.mean([res[1] for res in results])))  # Insert average overall score as first element of results

    # Restore original training values
    if issubclass(player.__class__, LearningPlayer):
        player.strategy.train, player.strategy.model.training = training_values

    if not silent:
        print("Overall score: %s" % results[0][1])

    return results[0][1], results


def evaluate_against_each_other(player1, player2):
    """
    Evaluates player1 vs player2 using direct matches in oder to determine which one is used as new reference player
    :param player1:
    :param player2:
    :return: True if player1 scores at least as high as player2
    """
    score, results = evaluate_against_base_players(player1, [player2])
    return score >= 0


def evaluate_both_players(player1, player2):
    """
    Evaluates both player2 and player1 against base players and each other to determine which one is used as new reference player.

    :param player1:
    :param player2:
    :return: True if player1 scores at least as high as player2
    """
    score, results = evaluate_against_base_players(player1, [player2])
    p1_score, results = evaluate_against_base_players(player1)
    p2_score, results = evaluate_against_base_players(player2)
    p1_score += score
    p2_score -= score

    return p1_score >= p2_score


def print_results(player, e_player, rewards):
    counter = Counter(rewards)
    try:
        lr = player.strategy.lr
    except AttributeError:
        lr = None

    print("\nEvaluating %s vs %s" % (player.__str__(), e_player.__str__()))
    print("Total score: %s" % np.mean(rewards))
    print("W/D/L: %s/%s/%s" % (counter[config.LABEL_WIN], counter[config.LABEL_DRAW], counter[config.LABEL_LOSS]))
