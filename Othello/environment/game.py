from two_player_game import TwoPlayerGame
import Othello.config as config
from Othello.environment.board import OthelloBoard


class Othello(TwoPlayerGame):

    def __init__(self, players, gui=None):
        super(Othello, self).__init__(players=players, config=config, gui=gui)

        self.player1.color = config.BLACK
        self.player2.color = config.WHITE

        for player in players:
            player.original_color = player.color

    def __run__(self, player1, player2):
        """
        Runs an episode of the game

        :param player1:
        :param player2:
        :return: The original color of the winning player
        """
        self.board = OthelloBoard()
        players = player1, player2
        if self.gui:
            self.gui.show_game(self.board)

        while True:
            if len(self.board.get_valid_moves(players[0].color)) > 0:
                move = players[0].get_move(self.board.copy())
                self.board.apply_move(move, players[0].color)

                if self.gui:
                    self.gui.flash_move(move, players[0].color)
                    self.gui.update(self.board, players[1])

                winner = self.board.game_won()
                if winner is not None:
                    if self.gui:
                        self.gui.show_winner(winner, self.board)
                    return config.get_label_from_winner_color(player1, player2, winner)

            players = list(reversed(players))

    def run_simulations(self, episodes, switch_colors=True, switch_players=True):
        """
        Runs a number of games using the given players and returns statistics over all games run.


        If both :param switch_colors and :param switch_players are set, all four possible starting positions will iterated through.
        :param episodes: The number of games to run
        :param switch_colors: Flag specifying whether to alternate the players colors during play
        :param switch_players: Flag specifying whether to alternate the starting player
        :return: The results and average losses per episode where results is a list of the original colors of the winning player ([original_winning_color])
        """

        simulation_players = [self.player1, self.player2]

        results = []
        losses = []

        for episode in range(episodes):
            if switch_colors and episode != 0 and episode % 2 == 0:
                simulation_players[0].color, simulation_players[1].color = simulation_players[1].color, simulation_players[0].color

            # Alternate starting color, effectively resulting in 4 starting positions rather than 2.
            if switch_players and episode != 0 and episode + 1 % 2:
                simulation_players = list(reversed(simulation_players))

            winner = self.__run__(simulation_players[0], simulation_players[1])
            player_losses = []
            for player in simulation_players:
                loss = player.register_winner(winner)
                if loss is not None:
                    player_losses.append(loss)

            losses += player_losses
            results.append(winner)

        for player in simulation_players:
            player.color = player.original_color

        return results, losses

