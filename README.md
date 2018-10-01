Deep Reinforcement Learning for TicTacToe and Othello
---------------------------------------------------------------
This repository contains all resources for my ongoing master's thesis on Deep Reinforcement Learning for zero sum games including TicTacToe and Othello.

I chose these games specifically because they represent a progression from a very simple game to a rather complicated one that can still be approached with resources available to an average student. I will release all my work here and it is my hope that other researchers will find both my framework as well as my players helpful and may even write their own players and or games to extend and contribute to it.

The thesis is available in pdf format and serves as documentation.

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

### Extending the Framework ###
When extending the framework please keep the original structure:
- Each game, it's environment and its players are kept in a separate directory nestled inside the project's main directory.
- Each of those directories contains subdirectories for **environment**, **experiments**, **players** and **tests** as well as a **config.py** file.
- There are abstract classes for **Board**, **Player**, **Strategy** and **Model** classes in the **AbstractClasses.py** file, please use them and add abstract base classes to it where necessary.
- A game's rules are contained in its **Board** (or equivalent) class, the **TwoPlayerGame** class does not contain any rules and can be reused for most two player games and extended for other types of games.

Simply put, add your player for an existing game to the game/players directory and the training script to game/experiments. If you want to add another game to the framework, create a new directory using a similar structure to the ones already available.

Acknowledgement
---------------
The framework I am using is based on [rgruener's Othello project](https://github.com/rgruener/othello).
