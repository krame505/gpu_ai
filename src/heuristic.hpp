#pragma once

#include "state.hpp"

#define HEURISTIC_SIGMA 0.11

#define CHECKER_VALUE      1
#define CHECKER_KING_VALUE 4

// Return the 'value' of a piece
#ifdef __CUDACC__
  __host__ __device__
#endif
  unsigned pieceValue(PieceType type);

// Calculate heuristic scores of a state for each player
#ifdef __CUDACC__
  __host__ __device__
#endif
  void scoreState(const State &state, unsigned score[NUM_PLAYERS]);

// Calculate the changes in heuristic scores of a move on a state for each player
#ifdef __CUDACC__
  __host__ __device__
#endif
  void scoreMove(const State &state, const Move &move, int score[NUM_PLAYERS]);

// Calculate the (unperturbed) weight for a new state from the scores of the current state and a move
#ifdef __CUDACC__
  __host__ __device__
#endif
  float getWeight(const State &state, unsigned stateScore[NUM_PLAYERS], int moveScore[NUM_PLAYERS]);
