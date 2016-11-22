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
  const int dr[] = {1, -1, 1, -1};
  const int dc[] = {-1, -1, 1, 1};
  // NOTE: player 1 is moving down, player 2 is moving up - fix if this assumption is wrong

  if (!(*this)[loc].occupied)
    return 0;

  if ((*this)[loc].type == CHECKER_KING) {
    for (uint8_t i = 0; i < 4; i++) {
      Move tmpMove(loc, Loc(loc.row + dr[i], loc.col + dc[i]));
      if (isValidMove(tmpMove))
        result[count++] = tmpMove;
    }
  } else {
    BoardItem item = (*this)[loc];
    uint8_t start = item.owner == PLAYER_1 ? 0 : 2;
    uint8_t end   = item.owner == PLAYER_1 ? 2 : 4;
    for (uint8_t i = start; i < end; i++) {
      Move tmpMove(loc, Loc(loc.row + dr[i], loc.col + dc[i]));
      if (isValidMove(tmpMove))
        result[count++] = tmpMove;
    }
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
  INIT_KERNEL_VARS

  __shared__ uint8_t indices[NUM_PLAYERS][NUM_LOCS];

  // Generate the moves for this location
  Move locMoves[MAX_LOC_MOVES];

  uint8_t numLocMoves = genLocDirectMoves(loc, locMoves);
  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
    if ((*this)[loc].owner == (PlayerId)i) {
      indices[i][id] = numLocMoves;
    }
    else {
      indices[i][id] = 0;
    }
  }

  // Perform exclusive scans to get indices to copy moves into the shared arrays
  for (uint8_t stride = 1; stride <= NUM_LOCS; stride <<= 1) {
    __syncthreads();
    uint8_t i = (id + 1) * stride - 1; // TODO: Check that this is correct...
    if (i < NUM_LOCS) {
      for (uint8_t j = 0; j < NUM_PLAYERS; j++) {
	indices[j][i] += indices[j][i - stride];
      }
    }
  }

  __syncthreads();

  if (id < NUM_PLAYERS) {
    numMoves[id] = indices[id][NUM_LOCS - 1];
    indices[id][NUM_LOCS - 1] = 0;
  }

  for (uint8_t stride = NUM_LOCS / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    int i = (id + 1) * stride - 1;
    uint8_t temp;
    for (uint8_t j = 0; j < NUM_PLAYERS; j++) {
      temp = indices[j][i];
      indices[j][i] += indices[j][i - stride];
      indices[j][i - stride] = temp;
    }
  }

  // Copy generated moves to shared arrays
  for (uint8_t i = 0; i < numLocMoves; i++) {
    result[turn][i + indices[turn][id]] = locMoves[i];
  }

#else
  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 0; j < BOARD_SIZE; j++) {
      Loc loc(i, j);
      PlayerId owner = (*this)[loc].owner;
      if (genMoves == NULL || genMoves[owner])
	numMoves[owner] += genLocDirectMoves(loc, &result[owner][numMoves[owner]]);
    }
  }

#endif
}

__host__ __device__ void State::genCaptureMoves(uint8_t numMoves[NUM_PLAYERS],
						Move result[NUM_PLAYERS][MAX_MOVES],
						bool genMoves[NUM_PLAYERS]) const {
#ifdef __CUDA_ARCH__
  // Do all the same stuff as genDirectMoves
  // Perform a reduction to calculate the max capture move length possible for each player
  // Each thread copies its capture moves of maximum length to a new local array
  // Perform another scan to calculate indices and copy the new arrays back to captureMoves

  INIT_KERNEL_VARS

    __shared__ uint8_t indices[NUM_PLAYERS][NUM_LOCS];

  // Generate the captures for this location
  Move locMoves[MAX_LOC_MOVES];

  uint8_t numLocCapture = genLocCaptureMoves(loc, locMoves);
  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
    if((*this)[loc].owner == (PlayerId)i) { 
      indices[i][id] = numLocCapture;
    }
    else {
      indices[i][id] = 0;
    }
  }

  // Perform exclusive scans to get indices to copy captures into the shared arrays
  for (uint8_t stride = 1; stride <= NUM_LOCS; stride <<= 1) {
    __syncthreads();
    uint8_t i = (id + 1) * stride - 1;
    if (i < NUM_LOCS) {
      for (uint8_t j = 0; j < NUM_PLAYERS; j++) {
	indices[j][i] += indices[j][i - stride];
      }
    }
  }

  __syncthreads();

  if (id < NUM_PLAYERS) {
    numMoves[id] = indices[id][NUM_LOCS - 1];
    indices[id][NUM_LOCS - 1] = 0;
  }

  for (uint8_t stride = NUM_LOCS/2; stride > 0; stride >>= 1) {
    __syncthreads();
    int i = (id + 1) * stride - 1;
    uint8_t temp;
    for (uint8_t j = 0; j < NUM_PLAYERS; j++) {
      temp = indices[j][i];
      indices[j][i] += indices[j][i - stride];
      indices[j][i - stride] = temp;
    }
  }

  // Copy generated captures to shared arrays
  for (uint8_t i = 0; i < numLocCapture; i++) {
    result[turn][i + indices[turn][id]] = locMoves[i];
  }


  // for (unsigned i = NUM_LOCS / 2; i > 0; i >>= 1) {
  //   __syncthreads();
  //   if (i > id) {
  //     for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
  //       if (captureMoveIndices[i][id + i] > captureMoveIndices[i][id])
  //         captureMoveIndices[i][id] = captureMoveIndices[i][id + i];
  //     }
  //   }
  // }

  // The number of capture moves for a location is 0 if there are any other locations with more capture moves
  // for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
  //   captureMoveIndices[i][id] = 0;
  // }
  // if (numLocCaptureMoves > numCaptureMoves[locOwner])
  //   numLocCaptureMoves = 0;
  // captureMoveIndices[locOwner][id] = numLocCaptureMoves;

  assert(false); // TODO

#else
  Move locMoves[BOARD_SIZE][BOARD_SIZE][MAX_MOVES];
  uint8_t locNumMoves[BOARD_SIZE][BOARD_SIZE];
  uint8_t maxJumps = 0;

  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 0; j < BOARD_SIZE; j++) {
      Loc loc(i, j);
      locNumMoves[i][j] = genLocDirectMoves(loc, locMoves[i][j]);
      for (int k = 0; k < locNumMoves[i][j]; k++) {
	if (locMoves[i][j][k].jumps > maxJumps)
	  locMoves[i][j][k].jumps = maxJumps;
      }
    }
  }

  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 0; j < BOARD_SIZE; j++) {
      Loc loc(i, j);
      PlayerId owner = (*this)[loc].owner;
      if (genMoves == NULL || genMoves[owner]) {
	uint8_t l = 0;
	for (uint8_t k = 0; k < locNumMoves[i][j]; k++) {
	  Move move = locMoves[i][j][k];
	  if (move.jumps == maxJumps) {
	    result[owner][l] = locMoves[i][j][k];
	    l++;
	  }
	}
      }
    }
  }

#endif
}

__host__ __device__ void State::genMoves(uint8_t numMoves[NUM_PLAYERS],
					 Move result[NUM_PLAYERS][MAX_MOVES],
					 bool genMoves[NUM_PLAYERS]) const {
#ifdef __CUDA_ARCH__
  __shared__ bool genMovesDefault[NUM_PLAYERS];
  if (threadIdx.x < NUM_PLAYERS) {
    genMovesDefault[threadIdx.x] = true;
  }
  
#else
  bool genMovesDefault[NUM_PLAYERS] = {true, true};
  
#endif

  if (!genMoves)
    genMoves = genMovesDefault;

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
