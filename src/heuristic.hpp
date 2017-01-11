#pragma once

#include "state.hpp"

#define HEURISTIC_SIGMA 0.1

// Return the 'value' of a piece
#ifdef __CUDACC__
  __host__ __device__
#endif
  unsigned pieceValue(PieceType type);

// Calculate a heuristic score of a state for the current player
#ifdef __CUDACC__
  __host__ __device__
#endif
  float scoreHeuristic(const State &state);
