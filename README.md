Deep Reinforcement Learning for TicTacToe, Connect4 and Othello
---------------------------------------------------------------
This repository contains all resources for my ongoing master's thesis on Deep Reinforcement Learning for zero sum games including TicTacToe, Connect4 and Othello.

I chose these games specifically because they represent a progression from a very simple game to a rather complicated one that can still be approached with resources available to an average student. I will release all my work here and it is my hope that other researchers will find both my framework as well as my players helpful and may even write their own players and or games to extend and contribute to it.

### Path to glory ###

This is a short outline for this project during my masters thesis.

1. __Tic Tac Toe__
    1. Build TicTacToe Environment with same structure as Othello
    2. Train simple Policy Gradient player for TicTacToe (2-3 Layer NN, lose when applying invalid move)
    3. Incorporate invalid moves into network - validate difference
    4. Try CNN on TicTacToe

2. __Go Bigger Go Better__
    1. Scale up TicTacToe board size and respective players network size to 8x8
    2. Apply the same players and framework to 7x6 Connect4 and 8x8 Othello

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

### Acknowledgement ###
The framework I am using is based on [rgruener's Othello project](https://github.com/rgruener/othello).
