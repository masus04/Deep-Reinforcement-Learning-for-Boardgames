import TicTacToe.config as config
from TicTacToe.environment.board import TicTacToeBoard


class TicTacToe:

    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

        self.player1.color = config.BLACK
        self.player2.color = config.WHITE
        for player in player1, player2:
            player.original_color = player.color

    def __run__(self, player1, player2):
        self.board = TicTacToeBoard()
        players = player1, player2

        while True:
            move = players[0].get_move(self.board)
            self.board.apply_move(move, players[0].color)

            winner = self.board.game_won()
            if winner:
                return player1 if player1.color == winner else player2

            players = list(reversed(players))

    def run_simulations(self, episodes, switch_colors=True, switch_players=True):
        simulation_players = [self.player1, self.player2]

        results = []

        for episode in range(episodes):
            if switch_colors and episode != 0 and episode % 2 == 0:
                simulation_players[0].color, simulation_players[1].color = simulation_players[1].color, simulation_players[0].color

            if switch_players and episode != 0 and episode + 1 % 2:
                simulation_players = list(reversed(simulation_players))

            winner = self.__run__(simulation_players[0], simulation_players[1])
            for player in simulation_players:
                player.register_winner(winner.original_color)

            results.append(winner.original_color)

        for player in simulation_players:
            player.color = player.original_color

        return results

