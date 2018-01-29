Path to glory
=============
This is a short outline of the third attempt at creating a Reinforcement Learning player for symmetric zero sum two player games.
This one features a clear path to follow. Let's see how far we get.

Phase One
---------
1. Build TicTacToe Environment with same structure as Othello
2. Train very simple Policy Gradient player for TicTacToe (2-3 Layer NN, lose when applying invalid move)
3. Incorporate invalid moves into network - validate difference
4. Try CNN on TicTacToe

Phase Two
---------
5. Scale up TicTacToe board size and respective players network size
6. Apply process to 4x4 Othello
7. Scale to 6x6 & 8x8 Othello
