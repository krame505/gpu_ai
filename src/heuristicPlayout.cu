
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

  // Calculate the scores for the initial state
  unsigned stateScore[NUM_PLAYERS];
  scoreState(state, stateScore);
  
  bool gameOver = false;

  do {
    uint8_t numMoves = state.genMovesParallel(moves);

    if (numMoves > 0) {
      Move optMove;
      int optMoveScore[NUM_PLAYERS] = {0, 0};
      float optWeight = -1/0.0; // -infinity

      // Calculate weights for each move and copy the best ones for each thread into an array
      for (uint8_t i = tx; i < numMoves; i += NUM_LOCS) {
	Move move = moves[i];
	int moveScore[NUM_PLAYERS] = {0, 0};
	scoreMove(state, move, moveScore);
	float weight = getWeight(state, stateScore, moveScore) + curand_normal(&generator) * HEURISTIC_SIGMA;
	if (i == tx || weight > optWeight) {
	  optMove = move;
	  optWeight = weight;
	  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	    optMoveScore[i] = moveScore[i];
	  }
	}
      }

      __shared__ float weights[NUM_LOCS];
      __shared__ float moveScores[NUM_LOCS][NUM_PLAYERS];
      if (tx < numMoves) {
	moves[tx] = optMove;
	weights[tx] = optWeight;
	for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	  moveScores[tx][i] = optMoveScore[i];
	}
      }
      else {
	weights[tx] = -1/0.0; // -infinity
      }

      // Perform reduction to find move with max weight
      for (uint8_t stride = NUM_LOCS / 2; stride > 0; stride >>= 1) {
	__syncthreads();
	if (tx < stride && weights[tx] < weights[tx + stride]) {
	  moves[tx] = moves[tx + stride];
	  weights[tx] = weights[tx + stride];
	  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	    moveScores[tx][i] = moveScores[tx + stride][i];
	  }
	}
      }
 
      if (tx == 0) {
	// Perform the move
	state.move(moves[0]);

	// Update the score for the current state
	for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	  stateScore[i] += moveScores[0][i];
	}
      }
    }
    else {
      gameOver = true; // No moves, game is over
    }
  } while (!gameOver);

  if (tx == 0)
    results[bx] = state.getNextTurn();
}

std::vector<PlayerId> DeviceHeuristicPlayoutDriver::runPlayouts(std::vector<State> states) {
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
