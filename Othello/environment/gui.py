import pygame
import sys
from pygame.locals import *
import time
from Othello.config import BLACK, WHITE
import os
from Othello.environment.menu import *


class Gui:

    def __init__(self):
        pygame.init()

        # colors
        self.BLACK = (0, 0, 0)
        self.BACKGROUND = (10, 10, 10)
        self.WHITE = (255, 255, 255)
        self.BLUE = (100, 100, 255)
        self.RED = (255, 50, 50)

        # display
        self.SCREEN_SIZE = (840, 480)
        self.BOARD_POS = (20, 20)
        self.BOARD = (40, 40)
        self.BOARD_SIZE = 400
        self.SQUARE_SIZE = 50
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)

        # fonts
        self.score_font = pygame.font.SysFont("Serif", 24)
        self.title_font = pygame.font.SysFont("Times New Roman", 34)
        self.font = pygame.font.Font(None, 32)

        # image files
        local_path = os.path.join(os.getcwd(), "..", "environment")
        self.board_img = pygame.image.load(os.path.join(local_path, "images", "board.bmp")).convert()
        self.black_img = pygame.image.load(os.path.join(local_path, "images", "preta.bmp")).convert()
        self.white_img = pygame.image.load(os.path.join(local_path, "images", "branca.bmp")).convert()
        self.tip_img = pygame.image.load(os.path.join(local_path, "images","tip.bmp")).convert()
        self.clear_img = pygame.image.load(os.path.join(local_path, "images","nada.bmp")).convert()

        pygame.display.set_caption('Othello Gym')
        # Ignore mouse motion (greatly reduces resources when not needed)
        pygame.event.set_blocked(pygame.MOUSEMOTION)

        self.ingame_menu = cMenu(100, 100, 20, 5, 'vertical', 100, self.screen,
                    [('Press S to Save Game', 1, None, None)])
        self.ingame_menu.set_center(True, True)
        self.ingame_menu.set_alignment('center', 'center')

    def show_game(self, board):
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.update(board)

    def show_winner(self, player_color, board):
        font = pygame.font.Font(None, 34)
        if player_color == WHITE:
            msg = font.render("White player wins", True, self.RED)
        elif player_color == BLACK:
            msg = font.render("Black player wins", True, self.RED)
        else:
            msg = font.render("Tie!", True, self.RED)
        msg2 = font.render("Press Mouse To Restart Game", True, self.RED)

        self.screen.blit(msg, msg.get_rect(centerx = self.screen.get_width()/2 - 175, centery = 225))
        self.screen.blit(msg2, msg2.get_rect(centerx = self.screen.get_width()/2 - 175, centery = 255))
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONDOWN:
                    return
                elif event.type == QUIT:
                    sys.exit(0)
            time.sleep(.05)

    def highlight_valid_moves(self, valid_moves):
        for move in valid_moves:
            move = (move[1], move[0])
            x = move[0]*self.SQUARE_SIZE + self.BOARD[0]
            y = move[1]*self.SQUARE_SIZE + self.BOARD[1]
            pygame.draw.rect(self.screen, (255, 215, 0), \
                    (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE), 1)
        pygame.display.flip()

    def flash_move(self, pos, color):
        for i in range(0, 3):
            self.put_stone(pos, color)
            time.sleep(.1)
            self.clear_square(pos)
            time.sleep(.1)
        self.put_stone(pos, color)
        time.sleep(.3)

    def put_stone(self, pos, color):
        """ draws piece with given position and color """
        if pos == None:
            return

        # flip orientation (because xy screen orientation)
        pos = (pos[1], pos[0])

        if color == BLACK:
            img = self.black_img
        elif color == WHITE:
            img = self.white_img
        else:
            img = self.clear_img

        x = pos[0]*self.SQUARE_SIZE + self.BOARD[0]
        y = pos[1]*self.SQUARE_SIZE + self.BOARD[1]

        self.screen.blit(img, (x, y), img.get_rect())
        pygame.display.flip()

    def clear_square (self, pos):
        # flip orientation
        pos =(pos[1], pos[0])

        x = pos[0]*self.SQUARE_SIZE + self.BOARD[0]
        y = pos[1]*self.SQUARE_SIZE + self.BOARD[1]
        self.screen.blit(self.clear_img,(x, y), self.clear_img.get_rect())
        pygame.display.flip()

    def get_move_by_mouse(self):
        state = 0
        prev_state = 1
        rect_list = []
        while True:
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONDOWN:
                    (mouse_x, mouse_y) = pygame.mouse.get_pos()
                    # Check to see if click was on the board
                    if mouse_x < self.BOARD_SIZE + self.BOARD[0] or \
                       mouse_x > self.BOARD[0] or \
                       mouse_y < self.BOARD_SIZE + self.BOARD[1] or \
                       mouse_y > self.BOARD[1] :
                        position = ((mouse_x - self.BOARD[0]) / self.SQUARE_SIZE), \
                                   ((mouse_y - self.BOARD[1]) / self.SQUARE_SIZE)
                        # flip orientation
                        position = (int(position[1]), int(position[0]))
                    return position
                if event.type == KEYDOWN:
                    if event.key == pygame.K_s:
                        self.save_board_to_file()
                elif event.type == QUIT:
                    sys.exit(0)
            pygame.display.flip()
            time.sleep(.05)

    def update(self, board, next_player=None):
        self.board = board
        self.player = next_player
        self.background.fill(self.BACKGROUND)
        self.screen.blit(self.background, (0, 0), self.background.get_rect())
        self.screen.blit(self.board_img, self.BOARD_POS, self.board_img.get_rect())
        for i in range(8):
            for j in range(8):
                if board.board[i][j] != 0:
                    self.put_stone(( i, j), board.board[i][j])

        self.showGameInformation(board, next_player)
        pygame.display.flip()

    def showGameInformation(self, board, next_player=None):
        title = self.title_font.render("Othello", True, self.WHITE, self.BACKGROUND)
        stones = board.count_stones()
        blacks_str = 'Black Pieces: %s ' % stones[0]
        whites_str = 'White Pieces: %s ' % stones[1]
        black_pieces_text = self.score_font.render(blacks_str, True, self.WHITE, self.BACKGROUND)
        white_pieces_text = self.score_font.render(whites_str, True, self.WHITE, self.BACKGROUND)

        if not next_player or next_player.color == BLACK:
            pygame.draw.circle(self.screen, (255, 215, 0), (550, 83), 7)
        else:
            pygame.draw.circle(self.screen, (255, 215, 0), (550, 123), 7)

        self.screen.blit(title, (600, 20))
        self.screen.blit(black_pieces_text,(565, 70))
        self.screen.blit(white_pieces_text,(565, 110))
