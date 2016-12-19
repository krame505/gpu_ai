#pragma once

// Note: __CUDACC__ macro tests are used to exclude __device__ and __host__ specifiers if compiling with gcc

#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include <iostream>

#define BOARD_SIZE 8
#define NUM_LOCS 32
#define NUM_PLAYERS 2

// TODO: Figure out actual values - these are probably overestimates
#define MAX_MOVE_JUMPS 8   // Max number of jumps that can be taken in a single move
#define MAX_LOC_MOVES  10  // Max number of possible moves for a piece from a single point
#define MAX_MOVES      100 // Max number of direct or capture moves possible

enum PlayerId {
  PLAYER_1,
  PLAYER_2,

  PLAYER_NONE=-1
};

enum PieceType {
  CHECKER,
  CHECKER_KING,
};

// Represents a location on a board
struct Loc {
  uint8_t row;
  uint8_t col;

#ifdef __CUDACC__
  __host__ __device__
#endif
  Loc(uint8_t row, uint8_t col) : row(row), col(col) {}

  // Default constructor must be empty to avoid a race condition when initializing shared memory
#ifdef __CUDACC__
  __host__ __device__
#endif
  Loc() {} //: row((uint8_t)-1), col((uint8_t)-1) {}

#ifdef __CUDACC__
  __host__ __device__
#endif
  bool operator==(const Loc &loc) const {
    return row == loc.row && col == loc.col;
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
  bool operator!=(const Loc &loc) const {
    return !((*this) == loc);
  }

  // Return true if this location is within the bounds of the board
#ifdef __CUDACC__
  __host__ __device__
#endif
  bool isValid() const;
};

// Represents the contents of the location on the board, either a piece or an empty squate
struct BoardItem {
  bool occupied;
  PieceType type;
  PlayerId owner;

  /*
#ifdef __CUDACC__
  __host__ __device__
#endif
  BoardItem(bool occupied, PieceType type, PlayerId owner) :
    occupied(occupied), type(type), owner(owner) {}

  // Default constructor must be empty to avoid a race condition when initializing shared memory
  // TODO: nvcc is claiming that this constructor is non-empty...
#ifdef __CUDACC__
  __host__ __device__
#endif
  BoardItem() {}*/
};

struct Move;

// Represents the current state of the game, including the board and current turn
struct State {
  BoardItem board[BOARD_SIZE][BOARD_SIZE];
  PlayerId turn;

#ifdef __CUDACC__
  __host__ __device__
#endif
  inline BoardItem *operator[](uint8_t row) {
    return &(board[row][0]);
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
  inline BoardItem operator[](Loc loc) const {
    return board[loc.row][loc.col];
  }

// Get the next player in the turn sequence
#ifdef __CUDACC__
  __host__ __device__
#endif
  PlayerId getNextTurn() const;

#ifdef __CUDACC__
  __host__ __device__
#endif
  void move(const Move &move);

  // Return true if the move is validly constructed and can be applied to the current state
#ifdef __CUDACC__
  __host__ __device__
#endif
  bool isValidMove(Move move) const;

  // Return true if the to location offset by deltaRow and deltaCol can be validly jumped
#ifdef __CUDACC__
  __host__ __device__
#endif
  bool isValidJump(Move move, Loc jumped, Loc newTo, bool checkCycles) const;

  // Generate the possible direct (i.e. not capture) moves from a location
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocDirectMoves(Loc, Move result[MAX_LOC_MOVES]) const;

  // Generate the possible capture moves of length 1 from a location
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocSingleCaptureMoves(Loc, Move result[MAX_LOC_MOVES]) const;

  // Generate the possible capture moves from a location
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocCaptureMoves(Loc, Move result[MAX_LOC_MOVES]) const;

private:
  // Helper to genLocSingleCaptureMoves, recursively generate the capture moves of length 1
  // for a regular checker
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocSingleCaptureReg(Loc, Move result[MAX_LOC_MOVES]) const;

  // Helper to genLocCaptureMoves, recursively generate the capture moves for a regular checker
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocCaptureReg(Loc, Move result[MAX_LOC_MOVES], uint8_t count=0, bool first=true) const;

  // Helper to genLocSingleCaptureMoves, recursively generate the capture moves of length 1
  // for a king
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocSingleCaptureKing(Loc, Move result[MAX_LOC_MOVES]) const;

  // Helper to genLocCaptureMoves, recursively generate the capture moves for a king
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocCaptureKing(Loc, Move result[MAX_LOC_MOVES], uint8_t count=0, bool first=true) const;

public:
  // Generate the possible moves from a location
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocMoves(Loc, Move result[MAX_LOC_MOVES]) const;

  // Generate all the possible capture or direct moves
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genTypeMoves(Move result[MAX_MOVES], bool isJump=false) const;

  // Generate all the possible moves
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genMoves(Move result[MAX_MOVES]) const;

  // Generate a vector of all the moves for the current turn
  std::vector<Move> getMoves() const;

  // Return true if the game is finished
  bool isGameOver() const;
};

// Represents a move, independant from any particular board state
struct Move {
  Loc from;
  Loc to;
  Loc removed[MAX_MOVE_JUMPS];
  Loc intermediate[MAX_MOVE_JUMPS];
  uint8_t jumps;
  bool promoted;

#ifdef __CUDACC__
  __host__ __device__
#endif
  Move(Loc from, Loc to, uint8_t jumps=0, bool promoted=false) :
    from(from), to(to), jumps(jumps), promoted(promoted) {}

  // Default constructor must be empty to avoid a race condition when initializing shared memory
#ifdef __CUDACC__
  __host__ __device__
#endif
  Move() {}

#ifdef __CUDACC__
  __host__ __device__
#endif
  bool operator==(const Move &other) const;

#ifdef __CUDACC__
  __host__ __device__
#endif
  bool operator!=(const Move &other) const {
    return !((*this) == other);
  }

  // add a jump to a new location - updates removed and intermediate steps
#ifdef __CUDACC__
  __host__ __device__
#endif
  void addJump(Loc newTo);

  // Return true if making this move prevents the other move from being made
#ifdef __CUDACC__
  __host__ __device__
#endif
  bool conflictsWith(const Move &other) const;
};

// Return the initial board setup
State getStartingState();

// Overload << operator for printing
std::ostream &operator<<(std::ostream&, PlayerId);
std::ostream &operator<<(std::ostream&, PieceType);
std::ostream &operator<<(std::ostream&, Loc);
std::ostream &operator<<(std::ostream&, BoardItem);
std::ostream &operator<<(std::ostream&, Move);
std::ostream &operator<<(std::ostream&, State);

// For debugging... gdb doesn't like calling overloaded operators
void printLoc(const Loc &loc);
void printMove(const Move &move);
void printState(const State &state);
