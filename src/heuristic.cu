
#include "heuristic.hpp"
#include "state.hpp"

#include <cassert>

#define HEURISTIC_THRESHOLD 1
#define MIN_VALUE 0.0001

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

__host__ __device__ PlayerId predictWinner(const State &state) {
#ifdef __CUDA_ARCH__
  uint8_t tx = threadIdx.x;
  uint8_t row = tx / (BOARD_SIZE / 2);
  uint8_t col = ((tx % (BOARD_SIZE / 2)) * 2) + (row % 2 == 0);
  Loc loc(row, col);

  __shared__ unsigned scores[NUM_LOCS][NUM_PLAYERS];
  
  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
    scores[tx][i] = 0;
  }

  if (state[loc].occupied)
    scores[tx][state[loc].owner] = pieceValue(state[loc].type);

  uint8_t stride = 1;
  while (stride < NUM_LOCS) {
    __syncthreads();
    if (((tx + 1) & ((stride << 1) - 1)) == 0) {
      for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	scores[tx][i] += scores[tx - stride][i];
      }
    }
    stride <<= 1;
  }
  __syncthreads();

  unsigned (&score)[NUM_PLAYERS] = scores[NUM_LOCS - 1];

#else
  unsigned score[NUM_PLAYERS] = {0, 0};
  for (uint8_t i = 0; i < BOARD_SIZE; i++) {
    for (uint8_t j = 1 - (i % 2); j < BOARD_SIZE; j+=2) {
      Loc loc(i, j);
      if (state[loc].occupied)
	score[state[loc].owner] += pieceValue(state[loc].type);
    }
  }

#endif

  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
    if (score[i] > score[(i + 1) % NUM_PLAYERS])
      return (PlayerId)i;
  }
  return PLAYER_NONE;
}
