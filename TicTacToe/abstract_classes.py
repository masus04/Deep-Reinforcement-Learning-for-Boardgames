class Board:

    def get_valid_moves(self, color):
        raise NotImplementedError("function |get_valid_moves| must be implemented by subclass")

    def apply_move(self, move, color):
        raise NotImplementedError("function |apply_move must| be implemented by subclass")

    def game_won(self):
        raise NotImplementedError("function |game_won| must be implemented by subclass")

    def get_representation(self, color):
        raise NotImplementedError("function |get_representation| must be implemented by subclass")

    def get_legal_moves_map(self, color):
        raise NotImplementedError("function |get_legal_moves_map| must be implemented by subclass")

    def copy(self):
        raise NotImplementedError("function |copy| must be implemented by subclass")

    def other_color(self, color):
        raise NotImplementedError("function |other_color| must be implemented by subclass")


class Player:

    color = None
    original_color = None

    def get_move(self, board):
        raise NotImplementedError("function |get_move| must be implemented by subclass")

    def register_winner(self, winner_color):
        pass

    def save(self):
        pass
