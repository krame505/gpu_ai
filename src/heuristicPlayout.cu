
#include "playout.hpp"
#include "state.hpp"
#include "heuristic.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>
#include <cassert>

#define CUDA_STACK_SIZE 1024 * 32

#define SEED 12345

__global__ void heuristicPlayoutKernel(State *states, PlayerId *results) {
  uint8_t tx = threadIdx.x;
  uint32_t bx = blockIdx.x;
  uint32_t tid = tx + (bx * NUM_LOCS);

  __shared__ State state;
  if (tx == 0)
    state = states[bx];

  // Init random generator
  curandState_t generator;
  curand_init(SEED, tid, 0, &generator);
 
  __shared__ Move moves[MAX_MOVES];
  __shared__ float moveScores[MAX_MOVES][NUM_PLAYERS];

  // Calculate the scores for the initial state
  unsigned stateScore[NUM_PLAYERS];
  scoreState(state, stateScore);

  while (1) {
    if (state.movesSinceLastCapture >= NUM_DRAW_MOVES) {
      if (tx == 0)
        results[bx] = PLAYER_NONE;
      break;
    }
    
    uint8_t numMoves = state.genMovesParallel(moves);
    if (numMoves == 0) {
      if (tx == 0)
        results[bx] = state.getNextTurn();
      break;
    }

    uint8_t optIndex;
    float optWeight = -1/0.0; // -infinity

    // Calculate weights for each move and copy the best ones for each thread into an array
    for (uint8_t i = tx; i < numMoves; i += NUM_LOCS) {
      Move move = moves[i];
      int moveScore[NUM_PLAYERS] = {0, 0};
      scoreMove(state, move, moveScore);
      float weight = getWeight(state, stateScore, moveScore) + curand_normal(&generator) * HEURISTIC_SIGMA;
      for (uint8_t j = 0; j < NUM_PLAYERS; j++) {
        moveScores[i][j] = moveScore[j];
      }
      if (weight > optWeight) {
        optIndex = i;
        optWeight = weight;
      }
    }

    __shared__ uint8_t indices[NUM_LOCS];
    __shared__ float weights[NUM_LOCS];
    if (tx < numMoves) {
      indices[tx] = optIndex;
      weights[tx] = optWeight;
    }
    else {
      indices[tx] = (uint8_t)-1;
      weights[tx] = -1/0.0; // -infinity
    }

    // Perform reduction to find move with max weight
    for (uint8_t stride = NUM_LOCS / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      if (tx < stride && weights[tx] < weights[tx + stride]) {
        indices[tx] = indices[tx + stride];
        weights[tx] = weights[tx + stride];
      }
    }
 
    uint8_t index = indices[0];

    // Perform the move
    if (tx == 0) {
      state.move(moves[index]);
    }

    // Update the score for the current state
    for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
      stateScore[i] += moveScores[index][i];
    }
  }
}

std::vector<PlayerId> DeviceHeuristicPlayoutDriver::runPlayouts(std::vector<State> states) {
  // If no playouts are being performed, return an empty vector to avoid launching an empty kernel
  if (states.empty()) {
    return std::vector<PlayerId>();
  }
  
  // Device variables
  State *devStates;
  PlayerId *devResults;
  cudaMalloc(&devStates, states.size() * sizeof(State));
  cudaMalloc(&devResults, states.size() * sizeof(PlayerId));

  // Copy states for playouts to device
  cudaMemcpy(devStates, states.data(), states.size() * sizeof(State), cudaMemcpyHostToDevice);

  // Increase default stack size
  cudaError_t error = cudaDeviceSetLimit(cudaLimitStackSize, CUDA_STACK_SIZE);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error setting stack size: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  // Invoke the kernel
  heuristicPlayoutKernel<<<states.size(), NUM_LOCS>>>(devStates, devResults);
  cudaDeviceSynchronize();

  // Check for errors
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error calling kernel: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  // Copy the results back to the host
  PlayerId results[states.size()];
  cudaMemcpy(results, devResults, states.size() * sizeof(PlayerId), cudaMemcpyDeviceToHost);

  // Free the global memory
  cudaFree(devStates);
  cudaFree(devResults);

  // Return a vector constructed from the contents of the array
  return std::vector<PlayerId>(results, results + states.size());
}
