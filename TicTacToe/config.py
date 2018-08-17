import torch
import os
from torch.autograd import Variable
from datetime import datetime

TIC_TAC_TOE_DIR = os.path.dirname(os.path.abspath(__file__))

# Encoding parameters
# Black > White because these are also used as rewards and for scoring.
BLACK = 1
WHITE = -1
EMPTY = 0

LABEL_WIN = BLACK
LABEL_LOSS = WHITE
LABEL_DRAW = EMPTY

# Board parameters
BOARD_SIZE = 3
WIN_LINE_LENGTH = 3

EVALUATION_GAMES = 20

# Network parameters
LR = 1e-5
GAMMA = 1  # 0.95   # Reward discounting factor
CLIP = 0.1          # Clipping parameter for PPO

CUDA = False  # torch.cuda.is_available() # Cuda is slower, probably because of small models and batch sizes


def get_color_from_player_number(code):
    if code == BLACK:
        return "Black"
    if code == WHITE:
        return "White"
    return "Empty"


def get_label_from_winner_color(player1, player2, winner):
    if player1.color == winner:
        return player1.original_color
    elif player2.color == winner:
        return player2.original_color
    else:
        return EMPTY


def time_diff(start):
    return str(datetime.now()-start).split(".")[0]


def make_variable(lst):
    var = Variable(torch.FloatTensor(lst))
    if CUDA:
        var = var.cuda(0)
    return var


def find_in_subdirectory(filename, subdirectory=''):
    if subdirectory:
        path = subdirectory
    else:
        path = os.getcwd()
    for root, dirs, names in os.walk(path):
        if filename in names:
            return os.path.join(root, filename)
    raise FileNotFoundError("File not found")
