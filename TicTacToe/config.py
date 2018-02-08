import torch
import os
from torch.autograd import Variable
from datetime import datetime

TIC_TAC_TOE_DIR = os.path.dirname(os.path.abspath(__file__))

# Encoding parameters
# Black > White because these are also used as rewards and for scoring.
BLACK = 1
WHITE = -1
EMPTY = 0.5

LABEL_WIN = BLACK
LABEL_LOSS = WHITE
LABEL_DRAW = EMPTY

# Board parameters
BOARD_SIZE = 3
WIN_LINE_LENGTH = 3

EVALUATION_GAMES = 40

# Network parameters
GAMMA = 0  # 0.99  # Reward discounting factor

CUDA = False  # torch.cuda.is_available() # Cuda is slower, probably because of small models and batch sizes


def get_color_from_player_number(code):
    if code == BLACK:
        return "Black"
    if code == WHITE:
        return "White"
    return "Empty"


def time_diff(start):
    return str(datetime.now()-start).split(".")[0]


def make_variable(lst):
    var = Variable(torch.FloatTensor(lst))
    if CUDA:
        var = var.cuda(0)
    return var


def findInSubdirectory(filename, subdirectory=''):
    if subdirectory:
        path = subdirectory
    else:
        path = os.getcwd()
    for root, dirs, names in os.walk(path):
        if filename in names:
            return os.path.join(root, filename)
    raise FileNotFoundError("File not found")
