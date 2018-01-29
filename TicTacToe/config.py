
BLACK = 1
WHITE = 2
EMPTY = 1.5

LABEL_WIN = BLACK
LABEL_LOSS = WHITE
LABEL_DRAW = EMPTY

BOARD_SIZE = 3
WIN_LINE_LENGTH = 3


def get_color_from_player_number(code):
    if code == BLACK:
        return "Black"
    if code == WHITE:
        return "White"
    return "Empty"
