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
  assert((*this)[move.from].occupied);
  assert(!(*this)[move.to].occupied);
  assert((*this)[move.from].owner == turn);

  board[move.to.row][move.to.col].occupied = true;
  board[move.to.row][move.to.col].type = move.newType;
  board[move.to.row][move.to.col].owner = turn;
  board[move.from.row][move.from.col].occupied = false;

  for (uint8_t i = 0; i < move.jumps; i++) {
    Loc removed = move.removed[i];

    // Error checking
    assert(removed.isValid());
    assert((*this)[removed].occupied);
    
    if (i < move.jumps - 1) {
      Loc intermediate = move.intermediate[i];
      assert(intermediate.isValid());
      assert(!(*this)[intermediate].occupied);
    }

    board[removed.row][removed.col].occupied = false;
  }

  turn = nextTurn(turn);
}


// This is taken from State::move but just returns false when an assertion would have failed
__host__ __device__ bool State::isValidMove(Move move) const {
  if (!move.to.isValid() || !move.from.isValid())
    return false;

  if (!(*this)[move.from].occupied || 
      (*this)[move.to].occupied || 
      (*this)[move.from].owner != turn ||
      move.from == move.to)
    return false;

  for (uint8_t i = 0; i < move.jumps; i++) {
    Loc removed = move.removed[i];
    if (!removed.isValid() ||
	!(*this)[removed].occupied ||
	(*this)[removed].owner == (*this)[move.from].owner)
      return false;
    
    if (i < move.jumps - 1) {
      Loc intermediate = move.intermediate[i];
      if (!intermediate.isValid() ||
	  (*this)[intermediate].occupied ||
          intermediate == move.to)
      return false;
    }
  }

  return true;
}


__host__ __device__ uint8_t State::genLocDirectMoves(Loc loc, Move result[MAX_LOC_MOVES]) const {
  uint8_t count = 0;
  const int dc[] = {1, -1, 1, -1};
  const int dr[] = {1, 1, -1, -1};
  // NOTE: player 1 is moving down, player 2 is moving up - fix if this assumption is wrong

  if (!(*this)[loc].occupied)
    return 0;

  if ((*this)[loc].type == CHECKER_KING) {
    for (uint8_t i = 0; i < 4; i++) {
      Move tmpMove(loc, Loc(loc.row + dr[i], loc.col + dc[i]), 0, false, CHECKER_KING);
      if (isValidMove(tmpMove))
        result[count++] = tmpMove;
    }
  } else {
    BoardItem item = (*this)[loc];
    uint8_t start = item.owner == PLAYER_1 ? 0 : 2;
    uint8_t end   = item.owner == PLAYER_1 ? 2 : 4;
    for (uint8_t i = start; i < end; i++) {
      Move tmpMove(loc, Loc(loc.row + dr[i], loc.col + dc[i]), 0, 
                   loc.row + dr[i] == (item.owner == PLAYER_1 ? (BOARD_SIZE - 1) : 0), 
                   (loc.row + dr[i] == (item.owner == PLAYER_1 ? (BOARD_SIZE - 1) : 0)) ? CHECKER_KING : CHECKER);
      if (isValidMove(tmpMove))
        result[count++] = tmpMove;
    }
  }

  return count;
}


__host__ __device__ uint8_t State::genLocCaptureMoves(Loc loc, Move result[MAX_LOC_MOVES]) const {
  if (!(*this)[loc].occupied)
    return 0;

  result[0] = Move(loc, loc);
  if ((*this)[loc].type == CHECKER)
    return genLocCaptureReg(loc, result);
  else
    return genLocCaptureKing(loc, result);
}


__device__ uint8_t State::genLocCaptureReg(Loc loc, Move result[MAX_LOC_MOVES], uint8_t count, bool first) const {

  // add an item if have jumped one piece already and it is the longest jump
  // possible (i.e. the set of jumps is not a proper subset of another possible
  // set of jumps) - can do with a DFS

  Loc toLeft, toRight;
  Move jumpToLeft = result[count], jumpToRight = result[count];

  if ((*this)[result[0].from].owner == PLAYER_1) {
    toLeft  = Loc(loc.row + 2, loc.col - 2);
    toRight = Loc(loc.row + 2, loc.col + 2);
  } else {
    toLeft  = Loc(loc.row - 2, loc.col - 2);
    toRight = Loc(loc.row - 2, loc.col + 2);
  }

  jumpToLeft.addJump(toLeft);
  jumpToRight.addJump(toRight);

  // no valid jumps but one was made at one point - save the move made at 
  // results[count] and indicate that it was successful by incrementing count
  if (!isValidMove(jumpToLeft) && !isValidMove(jumpToRight) && !first) {
    if ((*this)[result[0].from].owner == PLAYER_1 && result[count].to.row == (BOARD_SIZE - 1)) {
      result[count].promoted = true;
      result[count].newType = CHECKER_KING;
    }
    if ((*this)[result[0].from].owner == PLAYER_2 && result[count].to.row == 0) {
      result[count].promoted = true;
      result[count].newType = CHECKER_KING;
    }
    return count + 1;
  }

  // left branch is valid - at the very least save the jump in the result array
  // and explore this branch to see if more jumps can be made
  if (isValidMove(jumpToLeft)) {
    result[count] = jumpToLeft;
    count = genLocCaptureReg(toLeft, result, count, false);
  }

  // ditto for right branch
  if (isValidMove(jumpToRight)) {
    result[count] = jumpToRight;
    count = genLocCaptureReg(toRight, result, count, false);
  }

  return count;
}


__device__ uint8_t State::genLocCaptureKing(Loc loc, Move result[MAX_LOC_MOVES], uint8_t count, bool first) const {
  Loc locs[4] = { Loc(loc.row + 2, loc.col + 2), Loc(loc.row + 2, loc.col - 2),
                  Loc(loc.row - 2, loc.col + 2), Loc(loc.row - 2, loc.col - 2) };
  Move moves[4] = { result[count], result[count], result[count], result[count] };

  for (uint8_t i = 0; i < 4; i++)
    moves[i].addJump(locs[i]);

  if (!first &&
      !isValidMove(moves[0]) &&
      !isValidMove(moves[1]) &&
      !isValidMove(moves[2]) &&
      !isValidMove(moves[3])) {
    result[count].newType = CHECKER_KING;
    return count + 1;
  }

  for (uint8_t i = 0; i < 4; i++) {
    if (isValidMove(moves[i])) {
      result[count] = moves[i];
      count = genLocCaptureKing(locs[i], result, count, false);
    }
  }

  return count;
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

  if (id < NUM_PLAYERS && (genMoves == NULL || genMoves[id])) {
    numMoves[id] = indices[id][NUM_LOCS - 1];
    indices[id][NUM_LOCS - 1] = 0;
  }

  for (uint8_t stride = NUM_LOCS / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    int i = (id + 1) * stride - 1;
    uint8_t temp;
    for (uint8_t j = 0; j < NUM_PLAYERS; j++) {
      if (genMoves == NULL || genMoves[j]) {
	temp = indices[j][i];
	indices[j][i] += indices[j][i - stride];
	indices[j][i - stride] = temp;
      }
    }
  }

  // Copy generated moves to shared arrays
  for (uint8_t i = 0; i < numLocMoves; i++) {
    if (genMoves == NULL || genMoves[turn]) {
      result[turn][i + indices[turn][id]] = locMoves[i];
    }
  }

#else
  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
    if (genMoves == NULL || genMoves[i])
      numMoves[i] = 0;
  }

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

  if (id < NUM_PLAYERS && (genMoves == NULL || genMoves[id])) {
    numMoves[id] = indices[id][NUM_LOCS - 1];
    indices[id][NUM_LOCS - 1] = 0;
  }

  for (uint8_t stride = NUM_LOCS/2; stride > 0; stride >>= 1) {
    __syncthreads();
    int i = (id + 1) * stride - 1;
    uint8_t temp;
    for (uint8_t j = 0; j < NUM_PLAYERS; j++) {
      if (genMoves == NULL || genMoves[j]) {
	temp = indices[j][i];
	indices[j][i] += indices[j][i - stride];
	indices[j][i - stride] = temp;
      }
    }
  }

  // Copy generated captures to shared arrays
  for (uint8_t i = 0; i < numLocCapture; i++) {
    if (genMoves == NULL || genMoves[turn]) {
      result[turn][i + indices[turn][id]] = locMoves[i];
    }
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
      locNumMoves[i][j] = genLocCaptureMoves(loc, locMoves[i][j]);
      for (int k = 0; k < locNumMoves[i][j]; k++) {
	if (locMoves[i][j][k].jumps > maxJumps)
	  maxJumps = locMoves[i][j][k].jumps;
      }
    }
  }

  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
    if (genMoves == NULL || genMoves[i])
      numMoves[i] = 0;
  }

  uint8_t l = 0;
  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 0; j < BOARD_SIZE; j++) {
      Loc loc(i, j);
      PlayerId owner = (*this)[loc].owner;
      if (genMoves == NULL || genMoves[owner]) {
	for (uint8_t k = 0; k < locNumMoves[i][j]; k++) {
	  Move move = locMoves[i][j][k];
	  if (move.jumps == maxJumps) {
	    result[owner][l] = locMoves[i][j][k];
	    numMoves[owner]++;
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

__host__ __device__ bool Move::operator==(Move move) {
  if (from != move.from ||
      to != move.to ||
      jumps != move.jumps ||
      promoted != move.promoted ||
      newType != move.newType) {
    return false;
  }
  for (int i = 0; i < jumps; i++) {
    if (removed[i] != move.removed[i]) {
      return false;
    }
  }
  for (int i = 0; i < jumps - 1; i++) {
    if (intermediate[i] != move.intermediate[i]) {
      return false;
    }
  }
  return true;
}

__host__ __device__ void Move::addJump(Loc newTo) {
  if (jumps > 0)
    intermediate[jumps] = to;
  
  removed[jumps++] = Loc((newTo.row - to.row) / 2 + to.row,
			 (newTo.col - to.col) / 2 + to.col);
  to = newTo;
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

__host__ __device__ PlayerId State::result() const {
  //#ifdef __CUDA_ARCH__
  // TODO: Implement this seperately to make use of parallelism

  //#else
  int numPieces[NUM_PLAYERS] = {0, 0};
  int numKings[NUM_PLAYERS]  = {0, 0};

  for (int i = 0; i < BOARD_SIZE; i++) {
    for (int j = 0; j < BOARD_SIZE; j++) {
      if (board[i][j].occupied) {
	numPieces[board[i][j].owner]++;
	if (board[i][j].type == CHECKER_KING)
	  numKings[board[i][j].owner]++;
      }
    }
  }
  
  if (numPieces[PLAYER_1] == numPieces[PLAYER_2]) {
    if (numKings[PLAYER_1] == numKings[PLAYER_2])
      return PLAYER_NONE;
    else
      return numKings[PLAYER_1] > numKings[PLAYER_2] ? PLAYER_1 : PLAYER_2;
  } else {
    return numPieces[PLAYER_1] > numPieces[PLAYER_2] ? PLAYER_1 : PLAYER_2;
  }

  //#endif
}
