
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

    Loc playerOccupied[12];
    uint8_t numOccupied;
    bool noMoves[12];

    Move result[MAX_LOC_MOVES];
    
    do {
      numOccupied = 0;
      for (uint8_t i = 0; i < BOARD_SIZE; i++) {
	for (uint8_t j = 0; j < BOARD_SIZE; j++) {
	  Loc here(i, j);
	  if (state[here].occupied && state[here].owner == state.turn) {
	    playerOccupied[numOccupied] = here;
	    noMoves[numOccupied] = false;
	    numOccupied++;
	  }
	}
      }
      while (numOccupied > 0) {
	uint8_t tryLoc = (curand(&generator) % numOccupied) + 1;
	uint8_t index = 0;
	while (tryLoc > 0) {
	  while (noMoves[index] == true) {
	    index++;
	  }
	  tryLoc--;
	}
	uint8_t numMoves = state.genLocMoves(playerOccupied[index], result);
	if (numMoves == 0) {
	  numOccupied --;
	}
	else {
	  state.move(result[curand(&generator) % numMoves]);
	}
      }
      if (numOccupied == 0) {
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
