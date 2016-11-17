// Contains the implementation of functions that run on the host or device
#include "state.hpp"

// TODO: Set up some sort of error capture mechanism to check asserts in device code
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

__host__ __device__ void State::move(const Move &move) {
  // Error checking on host
#ifndef __CUDA_ARCH__
  move.to.assertValid();
  move.from.assertValid();
  assert(board[move.from.row][move.from.col].occupied);
  assert(!board[move.to.row][move.to.col].occupied);
#endif

  board[move.to.row][move.to.col] = (BoardItem){true, move.newType, move.player};
  board[move.from.row][move.from.col].occupied = false;
  for (uint8_t i = 0; i < move.jumps; i++) {
    Loc removed = move.removed[i];
#ifndef __CUDA_ARCH__
    Loc intermediate = move.intermediate[i];
    removed.assertValid();
    intermediate.assertValid();
    assert(board[removed.row][removed.col].occupied);
    assert(!board[intermediate.row][intermediate.col].occupied);
#endif

    board[removed.row][removed.col].occupied = false;
  }

  turn = nextTurn(turn);
}

__host__ __device__ bool State::isFinished() const {
  return false; // TODO
}

__host__ __device__ PlayerId State::result() const {
  //return PLAYER_NONE; // TODO
}

__host__ __device__ uint8_t State::locDirectMoves(Loc, Move result[MAX_LOC_MOVES]) const {
  //return 0; // TODO
}

__host__ __device__ uint8_t State::locCaptureMoves(Loc, Move result[MAX_LOC_MOVES]) const {
  //return 0; // TODO
}

__host__ __device__ uint8_t State::locMoves(Loc l, Move result[MAX_LOC_MOVES]) const {
  uint8_t numDirect = locDirectMoves(l, result);
  uint8_t numCapture = locCaptureMoves(l, result + numDirect);
  return numDirect + numCapture;
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

__host__ __device__ bool Move::conflictsWith(const Move &other) {
  //return false; // TODO
}
