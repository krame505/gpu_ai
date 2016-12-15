// Contains the implementation of functions that run on the host or device
// This file is compiled with nvcc

#include "state.hpp"

#include <assert.h>
#include <stdio.h>

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

  for (uint8_t i = 0; i < move.jumps; i++) {
    Loc removed = move.removed[i];

#ifndef NDEBUG
    // Error checking
    assert(removed.isValid());
    assert((*this)[removed].occupied);

    Loc intermediate = move.intermediate[i];
    assert(intermediate.isValid());
    assert(!(*this)[intermediate].occupied);
#endif

    board[removed.row][removed.col].occupied = false;
  }

  board[move.to.row][move.to.col].occupied = true;
  board[move.to.row][move.to.col].type = move.promoted ? CHECKER_KING : (*this)[move.from].type;
  board[move.to.row][move.to.col].owner = (*this)[move.from].owner;
  board[move.from.row][move.from.col].occupied = false;

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
    
    Loc intermediate = move.intermediate[i];
    if (!intermediate.isValid() ||
	(*this)[intermediate].occupied)
      return false;
  }

  return true;
}

__host__ __device__ bool State::isValidJump(Move move, Loc jumped, Loc newTo, bool checkCycles) const {
  if (!newTo.isValid() || !jumped.isValid())
    return false;

  assert(2 * jumped.row - move.to.row == newTo.row);
  assert(2 * jumped.col - move.to.col == newTo.col);

  if (!(*this)[jumped].occupied || 
      (*this)[jumped].owner == (*this)[move.from].owner ||
      (*this)[newTo].occupied)
    return false;

  if (checkCycles) {
    for (int8_t i = move.jumps - 1; i >= 0; i--) {
      if (newTo == move.intermediate[i])
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
      Loc to(loc.row + dr[i], loc.col + dc[i]);
      if (to.isValid() && !(*this)[to].occupied)
        result[count++] = Move(loc, to);
    }
  } else {
    BoardItem item = (*this)[loc];
    uint8_t start = item.owner == PLAYER_1 ? 0 : 2;
    uint8_t end   = item.owner == PLAYER_1 ? 2 : 4;
    for (uint8_t i = start; i < end; i++) {
      Loc to(loc.row + dr[i], loc.col + dc[i]);
      if (to.isValid() && !(*this)[to].occupied)
        result[count++] =
	  Move(loc, to, 0, 
	       loc.row + dr[i] == (item.owner == PLAYER_1 ? (BOARD_SIZE - 1) : 0));
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

  Move prevMove = result[count];

  int8_t deltaRowLeft, deltaColLeft, deltaRowRight, deltaColRight;
  if ((*this)[result[0].from].owner == PLAYER_1) {
    deltaRowLeft = 1;
    deltaColLeft = -1;
    deltaRowRight = 1;
    deltaColRight = 1;
  } else {
    deltaRowLeft = -1;
    deltaColLeft = -1;
    deltaRowRight = -1;
    deltaColRight = 1;
  }

  Loc jumpedLeft(loc.row + deltaRowLeft, loc.col + deltaColLeft);
  Loc jumpedRight(loc.row + deltaRowRight, loc.col + deltaColRight);
  Loc toLeft(loc.row + 2 * deltaRowLeft, loc.col + 2 * deltaColLeft);
  Loc toRight(loc.row + 2 * deltaRowRight, loc.col + 2 * deltaColRight);

  bool leftValid  = isValidJump(result[count], jumpedLeft, toLeft, false);
  bool rightValid = isValidJump(result[count], jumpedRight, toRight, false);

  // no valid jumps but one was made at one point - save the move made at 
  // results[count] and indicate that it was successful by incrementing count
  if (!leftValid && !rightValid && !first) {
    if ((*this)[result[0].from].owner == PLAYER_1 && result[count].to.row == (BOARD_SIZE - 1)) {
      result[count].promoted = true;
    }
    if ((*this)[result[0].from].owner == PLAYER_2 && result[count].to.row == 0) {
      result[count].promoted = true;
    }
    return count + 1;
  }

  // left branch is valid - at the very least save the jump in the result array
  // and explore this branch to see if more jumps can be made
  if (leftValid) {
    result[count] = prevMove;
    result[count].addJump(toLeft);
    count = genLocCaptureReg(toLeft, result, count, false);
  }

  // ditto for right branch
  if (rightValid) {
    result[count] = prevMove;
    result[count].addJump(toRight);
    count = genLocCaptureReg(toRight, result, count, false);
  }

  return count;
}


__device__ uint8_t State::genLocCaptureKing(Loc loc, Move result[MAX_LOC_MOVES], uint8_t count, bool first) const {
  Move prevMove = result[count];

  int8_t deltaRows[4] = {1, 1, -1, -1};
  int8_t deltaCols[4] = {1, -1, 1, -1};

  Loc toLocs[4];
  bool isValid[4];

  for (uint8_t i = 0; i < 4; i++) {
    Loc jumped(loc.row + deltaRows[i], loc.col + deltaCols[i]);
    toLocs[i] = Loc(loc.row + 2 * deltaRows[i], loc.col + 2 * deltaCols[i]);
    isValid[i] = isValidJump(prevMove, jumped, toLocs[i], true);
  }

  if (!first) {
    bool anyValid = false;
    for (uint8_t i = 0; i < 4; i++)
      anyValid |= isValid[i];
    if (!anyValid)
      return count + 1;
  }

  for (uint8_t i = 0; i < 4; i++) {
    if (isValid[i]) {
      result[count] = prevMove;
      result[count].addJump(toLocs[i]);
      count = genLocCaptureKing(toLocs[i], result, count, false);
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

__host__ __device__ void State::genTypeMoves(uint8_t numMoves[NUM_PLAYERS],
					     Move result[NUM_PLAYERS][MAX_MOVES],
					     bool genMoves[NUM_PLAYERS],
					     MoveType type) const {
#ifdef __CUDA_ARCH__
  INIT_KERNEL_VARS;
  
  __shared__ uint8_t indices[NUM_PLAYERS][NUM_LOCS];

  // Generate the moves for this location
  Move locMoves[MAX_LOC_MOVES];
  uint8_t numLocMoves;
  if (type == CAPTURE)
    numLocMoves = genLocCaptureMoves(loc, locMoves);
  else
    numLocMoves = genLocDirectMoves(loc, locMoves);
  
  PlayerId locOwner = (*this)[loc].owner;
  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
    if (locOwner == (PlayerId)i) {
      indices[i][id] = numLocMoves;
    }
    else {
      indices[i][id] = 0;
    }
  }

  // Reduce
  uint8_t stride = 1;
  while (stride < NUM_LOCS) {
    __syncthreads();
    if (((id + 1) & ((stride << 1) - 1)) == 0) {
      for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	indices[i][id] += indices[i][id - stride];
      }
    }
    stride <<= 1;
  }

  // Write zero to the last element after saving the value there as the block sum
  __syncthreads();
  if (id < NUM_PLAYERS && (genMoves == NULL || genMoves[id])) {
    numMoves[id] = indices[id][NUM_LOCS - 1];
    indices[id][NUM_LOCS - 1] = 0;
  }

  // Scan
  stride = NUM_LOCS / 2;
  while (stride > 0) {
    __syncthreads();
    uint8_t temp;
    if (((id + 1) & ((stride << 1) - 1)) == 0) {
      for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	if (genMoves == NULL || genMoves[i]) {
	  temp = indices[i][id - stride];
	  indices[i][id - stride] = indices[i][id];
	  indices[i][id] += temp;
	}
      }
    }
    stride >>= 1;
  }
  
  // Copy generated moves to shared arrays
  for (uint8_t i = 0; i < numLocMoves; i++) {
    if (genMoves == NULL || genMoves[locOwner]) {
      result[locOwner][i + indices[locOwner][id]] = locMoves[i];
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
      if (genMoves == NULL || genMoves[owner]) {
	if (type == CAPTURE)
	  numMoves[owner] += genLocCaptureMoves(loc, &result[owner][numMoves[owner]]);
	else
	  numMoves[owner] += genLocDirectMoves(loc, &result[owner][numMoves[owner]]);
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

  genTypeMoves(numMoves, result, genMoves, CAPTURE);

#ifdef __CUDA_ARCH__
  if (threadIdx.x < NUM_PLAYERS) {
    genMoves[threadIdx.x] &= !numMoves[threadIdx.x];
  }
#else
  for (int i = 0; i < NUM_PLAYERS; i++) {
    genMoves[i] &= !numMoves[i];
  }
#endif

  genTypeMoves(numMoves, result, genMoves, DIRECT);
}

__host__ __device__ bool Move::operator==(const Move &other) const {
  if (from != other.from ||
      to != other.to ||
      jumps != other.jumps ||
      promoted != other.promoted) {
    return false;
  }
  for (int i = 0; i < jumps; i++) {
    if (removed[i] != other.removed[i])
      return false;
    if (intermediate[i] != other.intermediate[i])
      return false;
  }
  return true;
}

__host__ __device__ void Move::addJump(Loc newTo) {
  assert(jumps < MAX_MOVE_JUMPS);
  intermediate[jumps] = newTo;
  removed[jumps++] = Loc((newTo.row - to.row) / 2 + to.row,
			 (newTo.col - to.col) / 2 + to.col);
  to = newTo;
}

__host__ __device__ bool Move::conflictsWith(const Move &other) const {
  // Cases of conflicting moves:

  // 1. two moves have the same destination
  if (to == other.to) return true;

  // 2. two moves capture the same piece.
  for (uint8_t i = 0; i < jumps; i++) {
    for (uint8_t j = 0; j < other.jumps; j++) {
      if (removed[i] == other.removed[j])
        return true;
    }
  }

  // 3. one piece ends up in the path of the other, or one piece captures the
  //    other (for moves of 2 different players)
  for (uint8_t i = 0; i < other.jumps; i++) {
    if (to == other.intermediate[i] || other.from == removed[i])
      return true;
  }

  // 4. TODO: other cases

  return false;
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
