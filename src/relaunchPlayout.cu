
#include "playout.hpp"
#include "state.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>

#define CUDA_STACK_SIZE 1024 * 32

#define SEED 12345

#define BLOCK_SIZE 32

__global__ void relaunchPlayoutKernel(State *states, PlayerId *results, int n, uint32_t *globaltempIndex) {
  uint8_t tx = threadIdx.x;
  uint32_t bx = blockIdx.x;
  uint32_t tid = tx + (bx * BLOCK_SIZE);
  uint32_t threadtempIndex = tid;


  if (tid < n) {
    State state;
    state = states[threadtempIndex];

    // Init random generator
    curandState_t generator;
    curand_init(SEED, threadtempIndex, 0, &generator);
 
    bool done = false;

    Move captureMoves[MAX_MOVES];
    Move directMoves[MAX_MOVES];
    uint8_t numMoveCapture, numMoveDirect;

    do {
      // Scan the board for pieces that can move
      numMoveCapture = 0;
      numMoveDirect = 0;
      for (uint8_t i = 0; i < BOARD_SIZE; i++) {
      	for (uint8_t j = 1 - (i % 2); j < BOARD_SIZE; j+=2) {
	        Loc here(i, j);
	        numMoveCapture += state.genLocSingleCaptureMoves(here, &captureMoves[numMoveCapture]);
	        numMoveDirect += state.genLocDirectMoves(here, &directMoves[numMoveDirect]);
	      }
      }

      if (numMoveCapture > 0) {
	      do {
	        uint8_t moveIndex = curand(&generator) % numMoveCapture;
	        Loc to = captureMoves[moveIndex].to;
	        state.move(captureMoves[moveIndex]);
	        state.turn = state.getNextTurn();
	        numMoveCapture = state.genLocSingleCaptureMoves(to, captureMoves);
	      } while (numMoveCapture > 0);
	        state.turn = state.getNextTurn();
      } 
      else if (numMoveDirect > 0) {
	      state.move(directMoves[curand(&generator) % numMoveDirect]);
      }
      else {
        results[threadtempIndex] = state.getNextTurn();
        if(*globaltempIndex >= n)
        	done = true;
        else{
          do {
          threadtempIndex = *globaltempIndex;
          } while (atomicCAS(globaltempIndex, threadtempIndex, *globaltempIndex + 1) != threadtempIndex);
           state = states[threadtempIndex]; 
        }
      }
    } while (!done);

  }
}

std::vector<PlayerId> DeviceRelaunchPlayoutDriver::runPlayouts(std::vector<State> states) {
  // Device variables
  State *devStates;
  PlayerId *devResults;
  cudaMalloc(&devStates, states.size() * sizeof(State));
  cudaMalloc(&devResults, states.size() * sizeof(PlayerId));
 
  uint32_t *globaltempIndex;
  cudaMalloc((void**) &globaltempIndex, sizeof(uint32_t));


  // Copy states for playouts to device
  cudaMemcpy(devStates, states.data(), states.size() * sizeof(State), cudaMemcpyHostToDevice);

  int numBlocks = 1024; //number of blocks  
  
  uint32_t temp = numBlocks * BLOCK_SIZE; //max number of threads that can run in parallel
  cudaMemcpy(globaltempIndex, &temp, sizeof(uint32_t), cudaMemcpyHostToDevice);

  cudaError_t error;

  // Invoke the kernel
  relaunchPlayoutKernel<<<numBlocks, BLOCK_SIZE>>>(devStates, devResults, states.size(), globaltempIndex);
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
