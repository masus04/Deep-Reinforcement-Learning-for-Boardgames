import unittest

import Othello.config as config
from Othello.environment.game import Othello
from Othello.experiments.othelloBaseExperiment import OthelloBaseExperiment
from Othello.players.reinforcePlayer import FCReinforcePlayer
from Othello.players.acPlayer import FCACPlayer
from Othello.players.ppoPlayer import FCPPOPlayer
from Othello.environment.evaluation import evaluate_against_base_players


class TestPlayers(unittest.TestCase):

    def test_ReinforcementPlayer(self):
        reinforce = FCReinforcePlayer(lr=1e-5)
        evaluate_against_base_players(reinforce, silent=False)

    def test_ACPlayer(self):
        actor_critic = FCACPlayer(lr=1e-5)
        evaluate_against_base_players(actor_critic, silent=False)

    def test_PPOPlayer(self):
        ppo = FCPPOPlayer(lr=1e-5)
        evaluate_against_base_players(ppo, silent=False)


if __name__ == '__main__':
    unittest.main()
