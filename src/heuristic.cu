
#include "heuristic.hpp"
#include "state.hpp"

#include <cassert>

__host__ __device__ unsigned pieceValue(PieceType type) {
  switch (type) {
  case CHECKER:
    return CHECKER_VALUE;
  case CHECKER_KING:
    return CHECKER_KING_VALUE;
  }

  assert(false);
  return 0;
}

__host__ __device__ void scoreState(const State &state, unsigned score[NUM_PLAYERS]) {
  for (uint8_t i = 0; i < NUM_PLAYERS; i++)
    score[i] = 0;
  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 1 - (i % 2); j < BOARD_SIZE; j+=2) {
      Loc loc(i, j);
      if (state[loc].occupied)
	score[state[loc].owner] += pieceValue(state[loc].type);
    }
  }
}

__host__ __device__ void scoreMove(const State &state,
				   const Move &move,
				   int score[NUM_PLAYERS]) {
  for (uint8_t i = 0; i < NUM_PLAYERS; i++)
    score[i] = 0;
  if (move.promoted)
    score[state.turn] += CHECKER_KING_VALUE - CHECKER_VALUE;
  for (uint8_t i = 0; i < move.jumps; i++) {
    score[state.getNextTurn()] -= pieceValue(state[move.removed[i]].type);
  }
}

__host__ __device__ float getWeight(const State &state,
				    unsigned stateScore[NUM_PLAYERS], 
				    int moveScore[NUM_PLAYERS]) {
  return (float)(stateScore[state.turn] + moveScore[state.turn]) /
    (stateScore[state.getNextTurn()] + moveScore[state.getNextTurn()]);
}
