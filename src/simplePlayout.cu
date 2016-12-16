
#include "playout.hpp"
#include "state.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>

#define CUDA_STACK_SIZE 1024 * 32

#define SEED 12345

__global__ void simplePlayoutKernel(State *states, PlayerId *results) {
  uint8_t tx = threadIdx.x;
  uint32_t bx = blockIdx.x;
  uint32_t tid = tx + (bx * NUM_LOCS);

  State state;
  state = states[tid];

  // Init random generator
  curandState_t generator;
  curand_init(SEED, tid, 0, &generator);
 
  bool gameOver = false;

  Loc playerOccupied[12];
  uint8_t numOccupied;
  
  do {
    numOccupied = 0;
    for (uint8_t i = 0; i < BOARD_SIZE; i++) {
      for (uint8_t j = 0; j < BOARD_SIZE; j++) {
	Loc here(i, j);
	if (state[here].occupied && state[here].owner == state.turn) {
	  playerOccupied[numOccupied++] = here;
	}
      }
    }
  } while (!gameOver);

  results[tid] = state.getNextTurn();
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

  // Invoke the kernel
  simplePlayoutKernel<<<states.size(), NUM_LOCS>>>(devStates, devResults);
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
