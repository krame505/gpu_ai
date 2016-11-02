#pragma once

// __CUDACC__ macro tests are needed to exclude __device__ and __host__ specifiers if compiling with gcc

#include <stdint.h>
#include <stdbool.h>
#include <vector>

#define BOARD_SIZE 8
#define MAX_REMOVED 8
#define NUM_PLAYERS 2

enum Player {
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
};

struct BoardItem {
  bool occupied;
  PieceType type;
  Player owner;
};

struct Move {
  Player player;
  union {
    struct {
      Loc from;
      Loc to;
      Loc removed[MAX_REMOVED];
    };
    Loc affected[];
  };
  uint8_t num_removed;
};

struct State {
  BoardItem board[BOARD_SIZE][BOARD_SIZE];
  Player turn;

#ifdef __CUDACC__
  __host__ __device__
#endif
  inline BoardItem *operator[](uint8_t row) {
    return &(board[row][0]);
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
  inline BoardItem operator[](Loc loc) {
    return board[loc.row][loc.col];
  }

  // TODO: Check passing reference on device is OK
  // TODO: Error checking?
#ifdef __CUDACC__
  __host__ __device__
#endif
 void move(const Move &move) {
    board[move.to.row][move.to.col] = board[move.from.row][move.from.col];
    board[move.from.row][move.from.col].occupied = false;
    for (uint8_t i = 0; i < move.num_removed; i++) {
      Loc removed = move.removed[i];
      board[removed.row][removed.col].occupied = false;
    }

    turn++;
    if (turn >= NUM_PLAYERS)
      turn = 0;
  }

  // Return true if the game is finished
#ifdef __CUDACC__
  __host__ __device__
#endif
  bool isFinished();

  // Return the winner, or NONE if a draw
#ifdef __CUDACC__
  __host__ __device__
#endif
  Player result();

  // TODO: There will be a device version of this as well...
  std::vector<Move> moves();
};

// Utility functions
#ifdef __CUDACC__
  __host__ __device__
#endif
Player nextTurn(Player);
