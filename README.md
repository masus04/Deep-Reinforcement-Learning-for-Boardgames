Path to glory
-------------
This is a short outline of the third attempt at creating a Reinforcement Learning player for symmetric zero sum two player games.
This one features a clear path to follow. Let's see how far we get.

1. __Tic Tac Toe__
    1. Build TicTacToe Environment with same structure as Othello
    2. Train very simple Policy Gradient player for TicTacToe (2-3 Layer NN, lose when applying invalid move)
    3. Incorporate invalid moves into network - validate difference
    4. Try CNN on TicTacToe

2. __Go Bigger Go Better__
    1. Scale up TicTacToe board size and respective players network size
    2. Apply process to 4x4 Othello
    3. Scale to 6x6 & 8x8 Othello


Operation Manual
----------------

### Installation ###
run the following in the root folder of the project in order to install required dependencies:
```
pip3 install
sudo apt-get install python3-tk
```

### Running experiments ###
For importing and file paths to work correctly always execute commands from the root directory of the project.

All files in `tests` and `experiments` folders are executable and create their artifacts in a folder next to the respective source file when executed in the following manner:
```
interpreter -m path.to.executable.file
```
As an example the environment tests can be started by typing the following in the root directory:
```
python3 -m TicTacToe.tests.testEnvironment
```
