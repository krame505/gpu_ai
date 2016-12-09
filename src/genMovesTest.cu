
#include "genMovesTest.hpp"
#include "state.hpp"

#include <iostream>
using namespace std;

__global__ void genMovesKernel(State *globalState, Move *globalMoves, uint8_t *globalNumMoves) {
  __shared__ State state;
  __shared__ uint8_t numMoves[NUM_PLAYERS];
  __shared__ Move moves[NUM_PLAYERS][MAX_MOVES];

  state = *globalState;
  state.genMoves(numMoves, moves);

  unsigned n, m;
  for (n = 0; n < NUM_PLAYERS; n ++) {
    globalNumMoves[n] = numMoves[n];
    for (m = 0; m < MAX_MOVES; m ++) {
      globalMoves[m + (n * MAX_MOVES)] = moves[n][m];
    }
  }
}

bool genMovesTest(State state) {
  // Device variables
  State *devState;
  Move *devMoves;
  uint8_t *devNumMoves;
  cudaMalloc(&devState, sizeof(State));
  cudaMalloc(&devMoves, NUM_PLAYERS * MAX_MOVES * sizeof(Move));
  cudaMalloc(&devNumMoves, NUM_PLAYERS * sizeof(uint8_t));

  // Copy states for playouts to device
  cudaMemcpy(devState, &state, sizeof(State), cudaMemcpyHostToDevice);

  // Invoke the kernel
  genMovesKernel<<<NUM_LOCS, 1>>>(devState, devMoves, devNumMoves);

  // Copy the results back to the host
  Move movesResult[NUM_PLAYERS * MAX_MOVES];
  cudaMemcpy(movesResult, devMoves, NUM_PLAYERS * MAX_MOVES * sizeof(Move), cudaMemcpyDeviceToHost);
  
  uint8_t numMovesResult[NUM_PLAYERS];
  cudaMemcpy(numMovesResult, devNumMoves, NUM_PLAYERS * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  uint8_t numMoves[NUM_PLAYERS];
  Move result[NUM_PLAYERS][MAX_MOVES];
  state.genMoves(numMoves, result);

  bool match = true;

  unsigned n, m;
  for (n = 0; n < NUM_PLAYERS; n ++) {
    if (numMovesResult[n] != numMoves[n]) {
      match = false;
      break;
    }
    for (m = 0; m < numMoves[n]; m ++) {
      if (movesResult[m + (n * MAX_MOVES)] != result[n][m]) {
        match = false;
        break;
      }
    }
  }

  if (!match) {
    cout << "Mismatch in CPU and GPU genMoves()" << endl;
    cout << state << endl;

    cout << "CPU Moves" << endl;
    for (n = 0; n < NUM_PLAYERS; n ++) {
      cout << "Player " << (n + 1) << " moves: " << (int)numMoves[n] << endl;
      for (m = 0; m < numMoves[n]; m ++) {
        cout << result[n][m] << endl;
      }
    }

    cout << "GPU Moves" << endl;
    for (n = 0; n < NUM_PLAYERS; n ++) {
      cout << "Player " << (n + 1) << " moves: " << (int)numMovesResult[n] << endl;
      for (m = 0; m < numMovesResult[n]; m ++) {
        cout << movesResult[m + (n * MAX_MOVES)] << endl;
      }
    }
  }

  return match;
}
