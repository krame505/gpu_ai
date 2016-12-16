
#include "playout.hpp"
#include "state.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>

#define CUDA_STACK_SIZE 1024 * 32

#define SEED 12345

#define BLOCK_SIZE 32

__global__ void simplePlayoutKernel(State *states, PlayerId *results, int n) {
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

    uint8_t numPieces, numCanMoveCapture, numCanMoveDirect;
    Move result[12][MAX_LOC_MOVES];
    uint8_t numMoves[12];
    uint8_t captureIndex[12];
    uint8_t directIndex[12];
    
    do {
      // Scan the board for pieces that can move
      numPieces = 0;
      numCanMoveCapture = 0;
      numCanMoveDirect = 0;
      for (uint8_t i = 0; i < BOARD_SIZE; i++) {
	for (uint8_t j = 0; j < BOARD_SIZE; j++) {
	  Loc here(i, j);
	  if (state[here].occupied && state[here].owner == state.turn) {
	    numMoves[numPieces] = state.genLocMoves(here, result[numPieces]);
	    if (numMoves[numPieces] > 0) {
	      if (result[numPieces][0].jumps > 0) {
		captureIndex[numCanMoveCapture] = numPieces;
		numCanMoveCapture++;
	      }
	      else {
		directIndex[numCanMoveDirect] = numPieces;
		numCanMoveDirect++;
	      }
	    }
	    numPieces++;
	  }
	}
      }

      if (numCanMoveCapture > 0) {
	uint8_t piece = captureIndex[curand(&generator) % numCanMoveCapture];
	uint8_t move = curand(&generator) % numMoves[piece];
	state.move(result[piece][move]);
      }
      else if (numCanMoveDirect > 0) {
	uint8_t piece = directIndex[curand(&generator) % numCanMoveDirect];
	uint8_t move = curand(&generator) % numMoves[piece];
	state.move(result[piece][move]);
      }
      else {
	gameOver = true;
      }
    } while (!gameOver);

    results[tid] = state.getNextTurn();
  }
}

std::vector<PlayerId> DeviceSimplePlayoutDriver::runPlayouts(std::vector<State> states) const {
  // Device variables
  State *devStates;
  PlayerId *devResults;
  cudaMalloc(&devStates, states.size() * sizeof(State));
  cudaMalloc(&devResults, states.size() * sizeof(PlayerId));

  // Copy states for playouts to device
  cudaMemcpy(devStates, states.data(), states.size() * sizeof(State), cudaMemcpyHostToDevice);

  cudaError_t error = cudaDeviceSetLimit(cudaLimitStackSize, CUDA_STACK_SIZE);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error setting stack size: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  int numBlocks = states.size() / BLOCK_SIZE;
  if (states.size() % BLOCK_SIZE)
    numBlocks++;
  
  // Invoke the kernel
  simplePlayoutKernel<<<numBlocks, BLOCK_SIZE>>>(devStates, devResults, states.size());
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

  // Return a vector constructed from the contents of the array
  return std::vector<PlayerId>(results, results + states.size());
}
