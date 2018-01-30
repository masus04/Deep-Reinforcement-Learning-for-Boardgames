import unittest

from TicTacToe.experiments.trainReinforcePlayer import TrainReinforcePlayer


class TestExperiments(unittest.TestCase):

    def test_trainReinforcePlayer(self):
        experiment = TrainReinforcePlayer().run(games=100, evaluations=10, lr=0.001)
        self.assertIsNotNone(experiment.last_plot)


if __name__ == '__main__':
    unittest.main()
