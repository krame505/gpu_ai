#pragma once

#include "state.hpp"

#ifdef __CUDACC__
  __host__ __device__
#endif
  unsigned pieceValue(PieceType type);

#ifdef __CUDACC__
  __host__ __device__
#endif
  PlayerId predictWinner(const State &state);
