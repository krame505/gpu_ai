#pragma once

// __CUDACC__ macro tests are needed to exclude __device__ and __host__ specifiers if compiling with gcc

#include <stdint.h>
#include <stdbool.h>
#include <vector>
#include <iostream>

#define BOARD_SIZE 8
#define NUM_LOCS 32
#define NUM_PLAYERS 2

// TODO: Figure out actual values - these are probably overestimates
#define MAX_MOVE_JUMPS 4   // Max number of jumps that can be taken in a single move
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

  // Assert that this location is within the bounds of the board
#ifdef __CUDACC__
  __host__ __device__
#endif
  void assertValid() const;
};

// Represents the contents of the location on the board, either a piece or an empty squate
struct BoardItem {
  bool occupied;
  PieceType type;
  PlayerId owner;

#ifdef __CUDACC__
  __host__ __device__
#endif
  BoardItem(bool occupied, PieceType type, PlayerId owner) :
    occupied(occupied), type(type), owner(owner) {}

  // Default constructor must be empty to avoid a race condition when initializing shared memory
#ifdef __CUDACC__
  __host__ __device__
#endif
  BoardItem() {}  
};

struct Move;
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

#ifdef __CUDACC__
  __host__ __device__
#endif
  void move(const Move &move);

  // Return true if the game is finished
#ifdef __CUDACC__
  __host__ __device__
#endif
  bool isFinished() const;

  // Return the winner, or NONE if a draw
#ifdef __CUDACC__
  __host__ __device__
#endif
  PlayerId result() const;

  // Generate the possible direct (i.e. not capture) moves from a location
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t locDirectMoves(Loc, Move[MAX_LOC_MOVES]) const;

  // Generate the possible capture moves from a location
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t locCaptureMoves(Loc, Move[MAX_LOC_MOVES]) const;

  // Generate the possible moves from a location
  // TODO: I don't think this ever will get called from the device?
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t locMoves(Loc, Move[MAX_LOC_MOVES]) const;

  std::vector<Move> moves() const;
};

struct Move {
  Loc from;
  Loc to;
  Loc removed[MAX_MOVE_JUMPS];
  Loc intermediate[MAX_MOVE_JUMPS];
  uint8_t jumps;
  bool promoted;
  PieceType newType; // Type after the move - same as original type if no promotion

#ifdef __CUDACC__
  __host__ __device__
#endif
  Move(Loc from, Loc to, uint8_t jumps, bool promoted=false, PieceType newType=CHECKER) :
    from(from), to(to), jumps(jumps), promoted(promoted), newType(newType) {}

  // Default constructor must be empty to avoid a race condition when initializing shared memory
#ifdef __CUDACC__
  __host__ __device__
#endif
  Move() {}

  // Return true if making this move prevents the other move from being made
#ifdef __CUDACC__
  __host__ __device__
#endif
  bool conflictsWith(const Move &other);
};

#ifdef __CUDACC__
  __host__ __device__
#endif
PlayerId nextTurn(PlayerId);

std::ostream &operator<<(std::ostream&, PlayerId);
std::ostream &operator<<(std::ostream&, PieceType);
std::ostream &operator<<(std::ostream&, Loc);
std::ostream &operator<<(std::ostream&, BoardItem);
std::ostream &operator<<(std::ostream&, Move);
std::ostream &operator<<(std::ostream&, State);
