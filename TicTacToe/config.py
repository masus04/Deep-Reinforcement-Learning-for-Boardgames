from doc_inherit_decorator import DocInherit

# Black > White because these are also used as rewards and for scoring.
BLACK = 2
WHITE = 1
EMPTY = 1.5

LABEL_WIN = BLACK
LABEL_LOSS = WHITE
LABEL_DRAW = EMPTY

BOARD_SIZE = 3
WIN_LINE_LENGTH = 3

EVALUATION_GAMES = 40


def get_color_from_player_number(code):
    if code == BLACK:
        return "Black"
    if code == WHITE:
        return "White"
    return "Empty"
