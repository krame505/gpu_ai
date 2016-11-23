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

    Loc from(7 - (input[1] - '1'), input[0] - 'a');
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

    if (i < len) {
      to = Loc(7 - (input[i + 1] - '1'), input[i] - 'a');
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
  // TODO
}
