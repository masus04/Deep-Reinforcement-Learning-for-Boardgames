from datetime import datetime
from TicTacToe.experiments.reinforced.crossValidationReinforced import ReinforcedCrossValidation
from TicTacToe.experiments.reinforced.trainACPlayer import TrainACPlayer


class ActorCriticCrossValidation(ReinforcedCrossValidation):
    pass


if __name__ == '__main__':

    start = datetime.now()

    GAMES = 10000
    EVALUATIONS = 100
    BATCH_SIZE = 1

    PLAYER = None  # PLAYER = Experiment.load_player("ReinforcePlayer using 3 layers pretrained on legal moves for 1000000 games.pth")

    experiment = ActorCriticCrossValidation(TrainACPlayer(games=GAMES, evaluations=EVALUATIONS, pretrained_player=PLAYER))

    results = experiment.run(5, -4.5, -5)

    print("\nFinal Reward - LR:")
    for res in results:
        print("%s - %s" % (res[0], res[1]))

    print("\nCrossvalidation complete, took: %s" % (datetime.now() - start))
