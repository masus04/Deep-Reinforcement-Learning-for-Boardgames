import unittest

from TicTacToe.experiments.train_reinforce_player import TrainReinforcePlayer


class TestExperiments(unittest.TestCase):

    def test_trainReinforcePlayer(self):
        experiment = TrainReinforcePlayer(100, 10).run()
        self.assertIsNotNone(experiment.last_plot)

    def test_fixCommandLineExecution(self):
        import sys;
        print(sys.executable)

        import os;
        print(os.getcwd())

        print(sys.path)


if __name__ == '__main__':
    unittest.main()
