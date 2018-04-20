import Connect4.config as config
from experiment import Experiment


class Connect4BaseExperiment(Experiment):

    def __init__(self):
        super(Connect4BaseExperiment, self).__init__(config)
