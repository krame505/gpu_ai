
#include "playout.hpp"
#include "state.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>

#define CUDA_STACK_SIZE 1024 * 32

#define SEED 12345

#define BLOCK_SIZE 32

__global__ void heuristicPlayoutKernel(State *states, PlayerId *results, int n) {
  uint8_t tx = threadIdx.x;
  uint32_t bx = blockIdx.x;
  uint32_t tid = tx + (bx * NUM_LOCS);

  if (tid < n) {
    State state;
    state = states[tid];

    // Init random generator
    curandState_t generator;
    curand_init(SEED, tid, 0, &generator);
 
    bool gameOver = false;

    Move captureMoves[MAX_MOVES];
    Move directMoves[MAX_MOVES];
    uint8_t numMoveCapture, numMoveDirect;
    float weight[MAX_MOVES];
    float totalWeight, p;
    
    do {
      // Scan the board for pieces that can move
      numMoveCapture = 0;
      numMoveDirect = 0;
      for (uint8_t i = 0; i < BOARD_SIZE; i++) {
	for (uint8_t j = 1 - (i % 2); j < BOARD_SIZE; j+=2) {
	  Loc here(i, j);
	  numMoveCapture += state.genLocCaptureMoves(here, &captureMoves[numMoveCapture]);
	  numMoveDirect += state.genLocDirectMoves(here, &directMoves[numMoveDirect]);
	}
      }

      totalWeight = 0.0f;
      p = curand_uniform(&generator);

      if (numMoveCapture > 0) {
	// Weight jumps higher if they capture more pieces, or if they are a regular checker that gets promoted
	for (uint8_t i = 0; i < numMoveCapture; i++) {
	  weight[i] = (float)captureMoves[i].jumps;
	  weight[i] += captureMoves[i].promoted ? 5.0f : 0.0f;
	  totalWeight += weight[i];
	}
	p *= totalWeight;
	for (uint8_t i = 0; i < numMoveCapture; i++) {
	  p -= weight[i];
	  if (p <= 0.0f) {
	    state.move(captureMoves[i]);
	    break;
	  }
	}	
      }
      else if (numMoveDirect > 0) {
	for (uint8_t i = 0; i < numMoveDirect; i++) {
	  // Weight checkers more highly if they are promoted or are more likely to be promoted
	  // Weight kings evenly
	  if (state[directMoves[i].from].type == CHECKER) {
	    if (state[directMoves[i].from].owner == PLAYER_1)
	      weight[i] = (float)directMoves[i].from.row;
	    else
	      weight[i] = 7.0f - (float)directMoves[i].from.row;
	    weight[i] += captureMoves[i].promoted ? 5.0f : 0.0f;
	  }
	  else {
	    weight[i] = 3.5f;
	  }
	  totalWeight += weight[i];
	}
	p *= totalWeight;
	for (uint8_t i = 0; i < numMoveDirect; i++) {
	  p -= weight[i];
	  if (p <= 0.0f) {
	    state.move(directMoves[i]);
	    break;
	  }
	}	
      }
      else {
	gameOver = true;
      }
    } while (!gameOver);

    results[tid] = state.getNextTurn();
  }
}

std::vector<PlayerId> DeviceHeuristicPlayoutDriver::runPlayouts(std::vector<State> states) {
  // Device variables
  State *devStates;
  PlayerId *devResults;
  cudaMalloc(&devStates, states.size() * sizeof(State));
  cudaMalloc(&devResults, states.size() * sizeof(PlayerId));

  // Copy states for playouts to device
  cudaMemcpy(devStates, states.data(), states.size() * sizeof(State), cudaMemcpyHostToDevice);

  int numBlocks = states.size() / BLOCK_SIZE;
  if (states.size() % BLOCK_SIZE)
    numBlocks++;

  cudaError_t error = cudaDeviceSetLimit(cudaLimitStackSize, CUDA_STACK_SIZE);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error setting stack size: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  // Invoke the kernel
  heuristicPlayoutKernel<<<numBlocks, BLOCK_SIZE>>>(devStates, devResults, states.size());
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
