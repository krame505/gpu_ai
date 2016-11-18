// Contains the implementation of functions that run on the host or device
// This file is compiled with nvcc

#include "state.hpp"

#include <assert.h>

__host__ __device__ bool Loc::isValid() const {
  return (row < BOARD_SIZE && col < BOARD_SIZE);
}

__host__ __device__ void State::move(const Move &move) {
  // Error checking
  assert(move.to.isValid());
  assert(move.from.isValid());
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
    assert(removed.isValid());
    assert(intermediate.isValid());
    assert(board[removed.row][removed.col].occupied);
    assert(!board[intermediate.row][intermediate.col].occupied);

    board[removed.row][removed.col].occupied = false;
  }

  turn = nextTurn(turn);
}


// This is taken from State::move but just returns false when an assertion would have failed
__host__ __device__ bool State::isValidMove(Move move) const {
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

__host__ __device__ uint8_t State::genLocDirectMoves(Loc loc, Move result[MAX_LOC_MOVES]) const {
  uint8_t count = 0;
  const int dx[] = {1, 1, -1, -1};
  const int dy[] = {1, -1, 1, -1};

  if (!(*this)[loc].occupied)
    return 0;

  if ((*this)[loc].type == CHECKER_KING) {
    for (uint8_t i = 0; i < 4; i++) {
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

__host__ __device__ uint8_t State::genLocCaptureMoves(Loc, Move result[MAX_LOC_MOVES]) const {
  //return 0; // TODO
}

__host__ __device__ uint8_t State::genLocMoves(Loc l, Move result[MAX_LOC_MOVES]) const {
  uint8_t numMoves = genLocCaptureMoves(l, result);
  if (numMoves == 0)
    numMoves = genLocDirectMoves(l, result);
  return numMoves;
}

__host__ __device__ void State::genDirectMoves(uint8_t numMoves[NUM_PLAYERS],
					       Move result[NUM_PLAYERS][MAX_MOVES],
					       bool genMoves[NUM_PLAYERS]) const {
#ifdef __CUDA_ARCH__

#else
  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 0; j < BOARD_SIZE; j++) {
      Loc loc(i, j);
      PlayerId owner = (*this)[loc].owner;
      if (!genMoves || genMoves[owner])
	numMoves[owner] += genLocDirectMoves(loc, &result[owner][numMoves[owner]]);
    }
  }

#endif
}

__host__ __device__ void State::genCaptureMoves(uint8_t numMoves[NUM_PLAYERS],
						Move result[NUM_PLAYERS][MAX_MOVES],
						bool genMoves[NUM_PLAYERS]) const {
#ifdef __CUDA_ARCH__

#else
  // TODO: Check capture moves have max length
  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 0; j < BOARD_SIZE; j++) {
      Loc loc(i, j);
      PlayerId owner = (*this)[loc].owner;
      if (!genMoves || genMoves[owner])
	numMoves[owner] += genLocDirectMoves(loc, &result[owner][numMoves[owner]]);
    }
  }

#endif
}

__host__ __device__ void State::genMoves(uint8_t numMoves[NUM_PLAYERS],
				      Move result[NUM_PLAYERS][MAX_MOVES],
				      bool genMoves[NUM_PLAYERS]) const {
  genCaptureMoves(numMoves, result, genMoves);

#ifdef __CUDA_ARCH__
  if (threadIdx.x < NUM_PLAYERS) {
    genMoves[threadIdx.x] &= !numMoves[threadIdx.x];
  }
#else
  for (int i = 0; i < NUM_PLAYERS; i++) {
    genMoves[i] &= !numMoves[i];
  }
#endif

  genDirectMoves(numMoves, result, genMoves);
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
