// Contains the implementation of functions that run on the host or device
// This file is compiled with nvcc

#include "state.hpp"

#include <assert.h>

__host__ __device__ void Loc::assertValid() const {
  assert(row < BOARD_SIZE);
  assert(col < BOARD_SIZE);
}

__host__ __device__ void State::move(const Move &move) {
  // Error checking
  move.to.assertValid();
  move.from.assertValid();
  assert(board[move.from.row][move.from.col].occupied);
  assert(!board[move.to.row][move.to.col].occupied);
  assert(board[move.from.row][move.from.col].owner == turn);

  board[move.to.row][move.to.col] =
    BoardItem(true, move.newType, board[move.from.row][move.from.col].owner);
  board[move.from.row][move.from.col].occupied = false;
  for (uint8_t i = 0; i < move.jumps; i++) {
    Loc removed = move.removed[i];

    // Error checking
    Loc intermediate = move.intermediate[i];
    removed.assertValid();
    intermediate.assertValid();
    assert(board[removed.row][removed.col].occupied);
    assert(!board[intermediate.row][intermediate.col].occupied);

    board[removed.row][removed.col].occupied = false;
  }

  turn = nextTurn(turn);
}


__host__ __device__ PlayerId State::result() const {
  //return PLAYER_NONE; // TODO
}


// This is taken from State::move but asserts just return false
__host__ __device__ bool State::isValidMove(Move move) const
{
  // Error checking
  if (!move.to.isValid() || !move.from.isValid())
    return false;

  if (!board[move.from.row][move.from.col].occupied || 
       board[move.to.row][move.to.col].occupied || 
       board[move.from.row][move.from.col].owner != turn)
    return false;

  for (uint8_t i = 0; i < move.jumps; i++) {
    Loc removed = move.removed[i];

    // Error checking
    Loc intermediate = move.intermediate[i];
    if (!removed.isValid() || !intermediate.isValid())
      return false;

    if (!board[removed.row][removed.col].occupied || 
         board[intermediate.row][intermediate.col].occupied)
      return false;
  }

  return true;
}

__host__ __device__ uint8_t State::locDirectMoves(Loc loc, Move result[MAX_LOC_MOVES]) const {
  uint8_t count = 0;
  const int dx[] = {1, 1, -1, -1};
  const int dy[] = {1, -1, 1, -1};

  if (!(*this)[loc].occupied)
    return 0;

  if ((*this)[loc].type == CHECKER_KING) {
    for(int i = 0; i < 4; i++) {
      Loc toLoc(loc.row + dx[i], loc.col + dy[i]);
      Move tmpMove(loc, toLoc);
      if (isValidMove(tmpMove))
        result[count++] = tmpMove;
    }
  } else {
    // TODO: add possible moves for non-king pieces - move depends on player
  }

  return count;
}

__host__ __device__ uint8_t State::locCaptureMoves(Loc, Move result[MAX_LOC_MOVES]) const {
  //return 0; // TODO
}

__host__ __device__ uint8_t State::locMoves(Loc l, Move result[MAX_LOC_MOVES]) const {
  uint8_t numDirect = locDirectMoves(l, result);
  uint8_t numCapture = locCaptureMoves(l, result + numDirect);
  return numDirect + numCapture;
}

__host__ __device__ bool Move::conflictsWith(const Move &other) {
  //return false; // TODO
}

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
