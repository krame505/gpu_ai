#include "player.hpp"

#include <string.h>

#include <vector>
#include <algorithm>
#include <random>
using namespace std;

Move RandomPlayer::getMove(const State &state) const {
  vector<Move> moves = state.getMoves();
  return moves[rand() % moves.size()];
}

Move HumanPlayer::getMove(const State &state) const {
  char input[100];

  while (true) {

    string error;
    vector<Move> moves = state.getMoves();

    cout << "Valid moves:" << endl;
    for (unsigned int n = 0; n < moves.size(); n ++)
    {
      cout << "Move " << n << ": " << moves[n] << endl;
    }

    cout << "Move for " << state.turn << ": ";
    cin.getline(input, 100);

    unsigned len = strlen(input);

    unsigned i = 0;
    Loc to(BOARD_SIZE, BOARD_SIZE); // Out of bounds

    if (len < 2 || len == 3 || len > 5) {
      error = "Invalid move syntax";
      moves.clear();
    }

    if (len > 2)
    {
      Loc from(BOARD_SIZE - (input[1] - '0'), input[0] - 'a');
      if (from.row >= BOARD_SIZE || from.col >= BOARD_SIZE) {
        error = "Invalid source location";
        moves.clear();
      }
      for (unsigned int n = 0; n < moves.size(); n ++) {
        if (moves[n].from.row != from.row || moves[n].from.col != from.col) {
          moves.erase(moves.begin() + n);
          n --;
        }
      }
      i += 2;
      if (input[i] == ' ')
        i++;
    }

    if (i < len) {
      to = Loc(BOARD_SIZE - (input[i + 1] - '0'), input[i] - 'a');
      if (to.row >= BOARD_SIZE || to.col >= BOARD_SIZE) {
        error = "Invalid destination";
        moves.clear();
      }
      for (unsigned int n = 0; n < moves.size(); n ++) {
        if (moves[n].to.row != to.row || moves[n].to.col != to.col) {
          moves.erase(moves.begin() + n);
          n --;
        }
      }
    }

    if (moves.size() == 0) {
      if (error.size())
        cout << error << ", try again" << endl;
      else
        cout << "Invalid move, try again" << endl;
    }
    else if (moves.size() > 1) {
      cout << "Ambiguous move" << endl;
    }
    else {
      return moves[0];
    }
  }
}

Move MCTSPlayer::getMove(const State &state) const {
  vector<Move> availableMoves = state.getMoves();
  vector<State> nextState;
  for (unsigned int n = 0; n < availableMoves.size(); n ++) {
    nextState[n] = state;
    nextState[n].move(availableMoves[n]);
  }

  vector<PlayerId> results = playouts(nextState);
  int numVictories = 0;
  for (unsigned int n = 0; n < results.size(); n ++) {
    // Check if the move is one that will lead to victory for this player
    if (state.turn == results[n]) {
      numVictories ++;
    }
  }

  if (numVictories > 0) {
    // Randomly pick from one of the victories (TODO: Pick the move that has the highest probability of victory?)
    int theMove = rand() % numVictories;
    for (unsigned int n = 0; n < results.size(); n ++)
    {
      if (state.turn == results[n]) {
        if (theMove == 0)
          return availableMoves[n];
        else
          theMove --;
      }
    }
  }

  // Otherwise just pick a random move
  return availableMoves[rand() % availableMoves.size()];
}
