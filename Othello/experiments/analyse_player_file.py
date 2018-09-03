import torch

import Othello.config as config
from Othello.experiments.othelloBaseExperiment import OthelloBaseExperiment


# LOAD_PLAYER = OthelloBaseExperiment.load_player("FCBaselinePlayer-Pretrained.pth")
LOAD_PLAYER = OthelloBaseExperiment.load_player("FCBaseline_trained_on_trad.pth")
COLOR = config.BLACK

game_states = OthelloBaseExperiment.generate_supervised_training_data(1, LOAD_PLAYER)

board = game_states[0][0]
board_var = config.make_variable(torch.FloatTensor([board.board]))
legal_moves = config.make_variable(board.get_legal_moves_map(COLOR))
probs, state_value = LOAD_PLAYER.strategy.model(board_var, legal_moves)
nonzero_move_probs = [p for p in probs.data[0] if p >= 1e-4]

print()
print("Number of legal moves: %s" % len(board.get_valid_moves(COLOR)))
print("Number of non zero move probs: %s" % len(nonzero_move_probs))
print(nonzero_move_probs)
print("State value: %s" % state_value.data[0][0])
print("Move probabilities: \n%s" % probs.data[0].view(8,8))

