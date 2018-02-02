from datetime import datetime

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


def get_color_from_player_number(code):
    if code == BLACK:
        return "Black"
    if code == WHITE:
        return "White"
    return "Empty"


def time_diff(start):
    return str(datetime.now()-start).split(".")[0]
