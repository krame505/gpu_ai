
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
  state = states[bx];

  // Init random generator
  curandState_t generator;
  curand_init(SEED, tid, 0, &generator);
 
  __shared__ Move moves[MAX_MOVES];

  bool gameOver = false;

  do {
    uint8_t numMoves = state.genMovesParallel(moves);

    if (numMoves > 0) {
      Move optMove;
      float optScore = -1/0.0; // -infinity

      // Calculate scores for each move and copy the best ones for each thread into an array
      for (uint8_t i = tx; i < numMoves; i += NUM_LOCS) {
	Move move = moves[i];
	State newState = state;
	newState.move(move);
	float score = scoreHeuristic(newState) + curand_normal(&generator) * HEURISTIC_SIGMA;
	if (i == tx || score > optScore) {
	  optMove = move;
	  optScore = score;
	}
      }

      __shared__ float scores[NUM_LOCS];
      if (tx < numMoves) {
	moves[tx] = optMove;
	scores[tx] = optScore;
      }
      else {
	scores[tx] = -1/0.0; // -infinity
      }

      // Perform reduction to find move with max score
      for (uint8_t stride = NUM_LOCS / 2; stride > 0; stride >>= 1) {
	__syncthreads();
	if (tx < stride && scores[tx] < scores[tx + stride]) {
	  moves[tx] = moves[tx + stride];
	  scores[tx] = scores[tx + stride];
	}
      }

      if (tx == 0) {
	// Perform the move
	state.move(moves[0]);
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
