
#include "state.hpp"

#include <cassert>
using namespace std;

__host__ __device__ PlayerId nextTurn(PlayerId turn) {
  switch (turn) {
  case PLAYER_1:
    return PLAYER_2;
  case PLAYER_2:
    return PLAYER_1;
  default:
    return PLAYER_NONE;
  }
}

__host__ __device__ bool Move::conflictsWith(const Move &other) {
  return false; // TODO
}

// TODO: Error checking on host
__host__ __device__ void State::move(const Move &move) {
  board[move.to.row][move.to.col] = board[move.from.row][move.from.col];
  board[move.from.row][move.from.col].occupied = false;
  for (uint8_t i = 0; i < move.jumps; i++) {
    Loc removed = move.removed[i];
    board[removed.row][removed.col].occupied = false;
  }

  turn = nextTurn(turn);
}

__host__ __device__ bool State::isFinished() const {
  return false; // TODO
}

__host__ __device__ PlayerId State::result() const {
  return PLAYER_NONE; // TODO
}

__host__ __device__ uint8_t State::locMoves(Loc, Move result[MAX_LOC_MOVES]) const {
  return 0; // TODO
}

std::vector<Move> State::moves() const {
  std::vector<Move> result;
  Move moves[MAX_LOC_MOVES];
  for (uint8_t row = 0; row < BOARD_SIZE; row++) {
    for (uint8_t col = 0; col < BOARD_SIZE; col++) {
      uint8_t numMoves = locMoves(Loc(row, col), moves);
      result.insert(result.end(), &moves[0], &moves[numMoves]);
    }
  }
  return result;
}

ostream &operator<<(ostream &os, PlayerId pi) {
  switch (pi) {
  case PLAYER_1:
    return os << "player 1";
  case PLAYER_2:
    return os << "player 1";
  default:
    assert(false);
    return os; // Unreachable, but to make -Wall shut up
  }
}

ostream &operator<<(ostream &os, PieceType pt) {
  switch (pt) {
  case PLAYER_1:
    return os << "checker";
  case PLAYER_2:
    return os << "king";
  default:
    assert(false);
    return os; // Unreachable, but to make -Wall shut up
  }
}

ostream &operator<<(ostream &os, Loc loc) {
  return os << string(1, 'a' + loc.col) << (loc.row + 1);
}

ostream &operator<<(ostream &os, Move m) {
  return os; // TODO
}

ostream &operator<<(ostream &os, State s) {
  return os; // TODO
}
