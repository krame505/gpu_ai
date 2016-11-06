#include "player.hpp"

#include <random>
using namespace std;

Move RandomPlayer::getMove(const State &state) const {
  vector<Move> moves = state.moves();
  return moves[rand() % moves.size()];
}

Move HumanPlayer::getMove(const State &state) const {
  vector<Move> moves = state.moves();
  return moves[0]; // TODO
}

Move MCTSPlayer::getMove(const State &state) const {
  vector<Move> moves = state.moves();
  return moves[0]; // TODO
}
