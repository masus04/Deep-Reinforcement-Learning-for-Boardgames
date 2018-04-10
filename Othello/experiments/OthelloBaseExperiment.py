from abc import ABC

import Othello.config as config
from experiment import Experiment


class OthelloBaseExperiment(Experiment):

    def __init__(self):
        super(OthelloBaseExperiment, self).__init__(config)
