class Board:

    def get_valid_moves(self, color):
        raise NotImplementedError("function get_move must be implemented by subclass")

    def apply_move(self, move, color):
        raise NotImplementedError("function get_move must be implemented by subclass")

    def game_won(self):
        raise NotImplementedError("function get_move must be implemented by subclass")

    def get_representation(self, color):
        raise NotImplementedError("function get_move must be implemented by subclass")

    def get_legal_moves_map(self, color):
        raise NotImplementedError("function get_move must be implemented by subclass")

    def copy(self):
        raise NotImplementedError("function get_move must be implemented by subclass")

    def other_color(self, color):
        raise NotImplementedError("function get_move must be implemented by subclass")
