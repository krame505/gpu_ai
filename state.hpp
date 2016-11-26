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
#define MAX_MOVE_JUMPS 4   // Max number of jumps that can be taken in a single move
#define MAX_LOC_MOVES  10  // Max number of possible moves for a piece from a single point
#define MAX_MOVES      100 // Max number of direct or capture moves possible

#define INIT_KERNEL_VARS				\
  uint8_t id = threadIdx.x;				\
  uint8_t row = id / (BOARD_SIZE / 2);			\
  uint8_t col = id % (BOARD_SIZE / 2) + (row % 2 == 0);	\
  Loc loc(row, col);


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

  // Return true if this location is within the bounds of the board 
#ifdef __CUDACC__
  __host__ __device__
#endif
  bool isValid() const;

#ifdef __CUDACC__
  __host__ __device__
#endif
  bool operator ==(Loc loc) {return row == loc.row && col == loc.col;}
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
  // TODO: nvcc is claiming that this constructor is non-empty...
#ifdef __CUDACC__
  __host__ __device__
#endif
  BoardItem() {}
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

#ifdef __CUDACC__
  __host__ __device__
#endif
  void move(const Move &move);

#ifdef __CUDACC__
  __host__ __device__
#endif
  bool isValidMove(Move move) const;

  // Generate the possible direct (i.e. not capture) moves from a location
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocDirectMoves(Loc, Move[MAX_LOC_MOVES]) const;

  // Generate the possible capture moves from a location
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocCaptureMoves(Loc, Move[MAX_LOC_MOVES]) const;



#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocCaptureReg(Loc, Move[MAX_LOC_MOVES], uint8_t count = 0, bool first = true) const;

#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocCaptureKing(Loc, Move[MAX_LOC_MOVES], uint8_t count = 0, bool first = true) const;




  // Generate the possible moves from a location
#ifdef __CUDACC__
  __host__ __device__
#endif
  uint8_t genLocMoves(Loc, Move[MAX_LOC_MOVES]) const;

  // Generate all the possible direct (i.e. not capture) moves for the given players
#ifdef __CUDACC__
  __host__ __device__
#endif
  void genDirectMoves(uint8_t[NUM_PLAYERS],
		      Move[NUM_PLAYERS][MAX_MOVES],
		      bool genMoves[NUM_PLAYERS] = NULL) const;

  // Generate all the possible capture moves
#ifdef __CUDACC__
  __host__ __device__
#endif
  void genCaptureMoves(uint8_t[NUM_PLAYERS],
		       Move[NUM_PLAYERS][MAX_MOVES],
		       bool genMoves[NUM_PLAYERS] = NULL) const;

  // Generate all the possible moves
#ifdef __CUDACC__
  __host__ __device__
#endif
  void genMoves(uint8_t[NUM_PLAYERS],
		Move[NUM_PLAYERS][MAX_MOVES],
		bool genMoves[NUM_PLAYERS] = NULL) const;

  // Generate a vector of all the moves for the current turn
  std::vector<Move> getMoves() const;

  // Return true if the game is finished
  bool isFinished() const;

  // Return the winner, or PLAYER_NONE if a draw
#ifdef __CUDACC__
  __host__ __device__
#endif
  PlayerId result() const;
};

// Represents a move, independant from any particular board state
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
  Move(Loc from, Loc to, uint8_t jumps=0, bool promoted=false, PieceType newType=CHECKER) :
    from(from), to(to), jumps(jumps), promoted(promoted), newType(newType) {}


  // add a jump to a new location - updates removed and intermediate steps
#ifdef __CUDACC__
  __host__ __device__
#endif
  void addJump(Loc newTo) {
    if (jumps > 1)
      intermediate[jumps] = to;

    removed[jumps++] = Loc((newTo.row - to.row) / 2 + to.row,
                           (newTo.col - to.col) / 2 + to.col);
    to = newTo;
  }

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

// Get the next player in the turn sequence
#ifdef __CUDACC__
  __host__ __device__
#endif
PlayerId nextTurn(PlayerId);

// Overload << operator for printing
std::ostream &operator<<(std::ostream&, PlayerId);
std::ostream &operator<<(std::ostream&, PieceType);
std::ostream &operator<<(std::ostream&, Loc);
std::ostream &operator<<(std::ostream&, BoardItem);
std::ostream &operator<<(std::ostream&, Move);
std::ostream &operator<<(std::ostream&, State);
