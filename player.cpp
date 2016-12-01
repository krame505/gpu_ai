
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
    for (unsigned int n = 0; n < moves.size(); n++) {
      cout << "Move " << n << ": " << moves[n] << endl;
    }

    cout << "Move for " << state.turn << ": ";
    cin.getline(input, 100);

    unsigned len = strlen(input);

    unsigned i = 0;
    Loc to(BOARD_SIZE, BOARD_SIZE); // Out of bounds

    if (len == 1 || len == 3 || len > 5) {
      error = "Invalid move syntax";
      moves.clear();
    }

    if (len > 2) {
      Loc from(BOARD_SIZE - (input[1] - '0'), input[0] - 'a');
      if (from.row >= BOARD_SIZE || from.col >= BOARD_SIZE) {
        error = "Invalid source location";
        moves.clear();
      }
      for (unsigned int n = 0; n < moves.size(); n++) {
        if (moves[n].from.row != from.row || moves[n].from.col != from.col) {
          moves.erase(moves.begin() + n);
          n--;
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
      for (unsigned int n = 0; n < moves.size(); n++) {
        if (moves[n].to.row != to.row || moves[n].to.col != to.col) {
          moves.erase(moves.begin() + n);
          n--;
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
  vector<unsigned> trials(MCTS_NUM_ITERATIONS, MCTS_NUM_TRIALS);
  GameTree *theTree = buildTree(state, trials);

  vector<GameTree*> children = theTree->getChildren();
  vector<Move> moves = theTree->getMoves();
  
  double highestScore = 0.0;
  unsigned int highestChild = 0;
    
  for (unsigned n = 0; n < children.size(); n++) {
    if (children[n]->getScore(state.turn) > highestScore) {
      highestScore = children[n]->getScore(state.turn);
      highestChild = n;
    }
  }

  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
    PlayerId player = (PlayerId)i;
    cout << player << " score: " <<
      children[highestChild]->getScore(player) << endl;
  }

  delete theTree;
  
  return moves[highestChild];
}
