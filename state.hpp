#pragma once

#include <stdint.h>
#include <stdbool.h>

#define BOARD_SIZE 8
#define MAX_REMOVED 8

enum Player {
  PLAYER_1,
  PLAYER_2,
};

enum PieceType {
  CHECKER,
  CHECKER_KING,
};

struct Loc {
  uint8_t row;
  uint8_t col;
}

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
}

struct State {
  BoardItem board[BOARD_SIZE][BOARD_SIZE];
  Player turn;

  __host__ __device__ inline BoardItem *operator[](uint8_t row) {
    return &(board[row]);
  }

  __host__ __device__ inline BoardItem operator[](Loc loc) {
    return board;
  }

  // TODO: Check passing reference on device is OK
  __host__ __device__ move(const Move &move) {
    board[move.to.row][move.to.col] = board[move.from];
    board[move.from].occupied = false;
    for (uint8_t i = 0; i < num_removed; i++) {
      board[move.removed[i]].occupied = false;
    }
  }
};
