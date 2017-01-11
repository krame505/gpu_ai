
#include "heuristic.hpp"
#include "state.hpp"

#include <cassert>

__host__ __device__ unsigned pieceValue(PieceType type) {
  switch (type) {
  case CHECKER:
    return 1;
  case CHECKER_KING:
    return 4;
  }

  assert(false);
  return 0;
}

__host__ __device__ float scoreHeuristic(const State &state) {
  unsigned score[NUM_PLAYERS] = {0, 0};
  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 1 - (i % 2); j < BOARD_SIZE; j+=2) {
      Loc loc(i, j);
      if (state[loc].occupied)
	score[state[loc].owner] += pieceValue(state[loc].type);
    }
  }

  return (float)score[state.turn] / score[state.getNextTurn()];
}
