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

  cout << "Move for " << state.turn << ": ";
  while (true) {
    cin.getline(input, 100);

    string error;
    vector<Move> moves = state.getMoves();
    unsigned len = strlen(input);

    unsigned i = 0;
    Loc to(BOARD_SIZE, BOARD_SIZE); // Out of bounds

    if (len < 2 || len == 3 || len > 5) {
      error = "Invalid move syntax";
      moves.clear();
    }

    if (len > 2) {
      Loc from(input[1] - '1', input[0] - 'a');
      if (from.row >= BOARD_SIZE || from.col >= BOARD_SIZE) {
        error = "Invalid source location";
        moves.clear();
      }
      // TODO: Check if intermediate locations are open when doing jump moves 
      moves.erase(remove_if(moves.begin(), moves.end(),
                            [state, from](Move m) {
                              if (state[from].occupied) {
                                State state2 = state;
                                state2.move(m);
                                return !state2[from].occupied;
                              }
                              return true;
                            }),
                  moves.end());
      i += 2;
      if (input[i] == ' ')
        i++;
    }

    if (i < len) {
      to = Loc(input[i + 1] - '1', input[i] - 'a');
      if (to.row >= BOARD_SIZE || to.col >= BOARD_SIZE) {
        error = "Invalid destination";
        moves.clear();
      }
      moves.erase(remove_if(moves.begin(), moves.end(),
                            [state, to](Move m) {
                              if (!state[to].occupied) {
                                State state2 = state;
                                state2.move(m);
                                return state2[to].occupied;
                              }
                              return true;
                            }),
                  moves.end());
    }

    if (moves.size() == 0) {
      if (error.size())
        cout << error << ", try again: ";
      else
        cout << "Invalid move, try again: ";
    }
    else if (moves.size() > 1) {
      cout << "Ambiguous move: ";
    }
    else {
      return moves[0];
    }
  }
}

Move MCTSPlayer::getMove(const State &state) const {
  // TODO
}
