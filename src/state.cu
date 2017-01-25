// Contains the implementation of functions that run on the host or device
// This file is compiled with nvcc

#include "state.hpp"

#include <assert.h>
#include <stdio.h>

__host__ __device__ bool Loc::isValid() const {
  return (row < BOARD_SIZE && col < BOARD_SIZE);
}

__host__ __device__ PlayerId State::getNextTurn() const {
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

  turn = getNextTurn();
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
      (*this)[jumped].owner == turn ||
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

__host__ __device__ uint8_t State::genLocMoves(Loc loc, Move result[MAX_LOC_MOVES], MoveType type) const {
  if (!(*this)[loc].occupied || (*this)[loc].owner != turn)
    return 0;

  switch (type) {
  case DIRECT:
    return genLocDirectMoves(loc, result);

  case CAPTURE:
    result[0] = Move(loc, loc);
    if ((*this)[loc].type == CHECKER)
      return genLocCaptureReg(loc, result);
    else
      return genLocCaptureKing(loc, result);

  case SINGLE_CAPTURE:
    if ((*this)[loc].type == CHECKER)
      return genLocSingleCaptureReg(loc, result);
    else
      return genLocSingleCaptureKing(loc, result);
  }

  // Should never get here, can't throw an exception in device code
  assert(false);
  return 0;
}

__host__ __device__ uint8_t State::genTypeMoves(Move result[MAX_MOVES], MoveType type) const {
  uint8_t numMoves = 0;
  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 1 - (i % 2); j < BOARD_SIZE; j+=2) {
      Loc loc(i, j);
      numMoves += genLocMoves(loc, &result[numMoves], type);
    }
  }
  return numMoves;
}

__device__ uint8_t State::genTypeMovesParallel(Move result[MAX_MOVES], MoveType type) const {
  assert(blockDim.x == NUM_LOCS);

  uint8_t tx = threadIdx.x;
  uint8_t row = tx / (BOARD_SIZE / 2);
  uint8_t col = ((tx % (BOARD_SIZE / 2)) * 2) + (row % 2 == 0);
  Loc loc(row, col);

  uint8_t numMoves = 0;
  
  __shared__ uint8_t indices[NUM_LOCS];

  // Generate the moves for this location
  Move locMoves[MAX_LOC_MOVES];
  uint8_t numLocMoves = genLocMoves(loc, locMoves, type);
  indices[tx] = numLocMoves;

  // Reduce
  uint8_t stride = 1;
  while (stride < NUM_LOCS) {
    __syncthreads();
    if (((tx + 1) & ((stride << 1) - 1)) == 0) {
      indices[tx] += indices[tx - stride];
    }
    stride <<= 1;
  }

  // Write zero to the last element after saving the value there as numMoves
  __syncthreads();
  numMoves = indices[NUM_LOCS - 1];
  __syncthreads();
  if (tx == 0) {
    indices[NUM_LOCS - 1] = 0;
  }

  // Scan
  stride = NUM_LOCS / 2;
  while (stride > 0) {
    __syncthreads();
    if (((tx + 1) & ((stride << 1) - 1)) == 0) {
      uint8_t temp = indices[tx - stride];
      indices[tx - stride] = indices[tx];
      indices[tx] += temp;
    }
    stride >>= 1;
  }
  
  // Copy generated moves to results array
  for (uint8_t i = 0; i < numLocMoves; i++) {
    result[i + indices[tx]] = locMoves[i];
  }

  __syncthreads();

  return numMoves;
}

__host__ __device__ uint8_t State::genMoves(Move result[MAX_MOVES]) const {
  uint8_t numMoves = genTypeMoves(result, CAPTURE);
  if (numMoves == 0)
    numMoves = genTypeMoves(result, DIRECT);

  return numMoves;
}

__device__ uint8_t State::genMovesParallel(Move result[MAX_MOVES]) const {
  uint8_t numMoves = genTypeMovesParallel(result, CAPTURE);
  if (numMoves == 0)
    numMoves = genTypeMovesParallel(result, DIRECT);

  return numMoves;
}

__host__ __device__ uint8_t State::genLocDirectMoves(Loc loc, Move result[MAX_LOC_MOVES]) const {
  uint8_t count = 0;
  const int dc[] = {1, -1, 1, -1};
  const int dr[] = {1, 1, -1, -1};
  // NOTE: player 1 is moving down, player 2 is moving up - fix if this assumption is wrong

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

__host__ __device__ uint8_t State::genLocCaptureReg(Loc loc, Move result[MAX_LOC_MOVES], uint8_t count, bool first) const {

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
    if (turn == PLAYER_1 && result[count].to.row == (BOARD_SIZE - 1)) {
      result[count].promoted = true;
    }
    if (turn == PLAYER_2 && result[count].to.row == 0) {
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

__host__ __device__ uint8_t State::genLocSingleCaptureReg(Loc loc, Move result[MAX_LOC_MOVES]) const {
  uint8_t count = 0;
  
  int8_t deltaRowLeft, deltaColLeft, deltaRowRight, deltaColRight;
  if ((*this)[loc].owner == PLAYER_1) {
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

  Loc toLeft(loc.row + 2 * deltaRowLeft, loc.col + 2 * deltaColLeft);
  Loc toRight(loc.row + 2 * deltaRowRight, loc.col + 2 * deltaColRight);

  Move moveLeft(loc, loc, 0, false);
  moveLeft.addJump(toLeft);
  if (isValidMove(moveLeft)) {
    if (turn == PLAYER_1 && moveLeft.to.row == (BOARD_SIZE - 1)) {
      moveLeft.promoted = true;
    }
    if (turn == PLAYER_2 && moveLeft.to.row == 0) {
      moveLeft.promoted = true;
    }
    result[count++] = moveLeft;
  }

  Move moveRight(loc, loc, 0, false);
  moveRight.addJump(toRight);
  if (isValidMove(moveRight)) {
    if (turn == PLAYER_1 && moveRight.to.row == (BOARD_SIZE - 1)) {
      moveRight.promoted = true;
    }
    if (turn == PLAYER_2 && moveRight.to.row == 0) {
      moveRight.promoted = true;
    }
    result[count++] = moveRight;
  }

  return count;
}

__host__ __device__ uint8_t State::genLocCaptureKing(Loc loc, Move result[MAX_LOC_MOVES], uint8_t count, bool first) const {
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

__host__ __device__ uint8_t State::genLocSingleCaptureKing(Loc loc, Move result[MAX_LOC_MOVES]) const {
  uint8_t count = 0;
  
  int8_t deltaRows[4] = {1, 1, -1, -1};
  int8_t deltaCols[4] = {1, -1, 1, -1};

  for (uint8_t i = 0; i < 4; i++) {
    Loc toLoc = Loc(loc.row + 2 * deltaRows[i], loc.col + 2 * deltaCols[i]);
    Move move(loc, loc, 0, false);
    move.addJump(toLoc);
    if (isValidMove(move)) {
      result[count++] = move;
    }
  }

  return count;
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
