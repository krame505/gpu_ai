
#include "playout.hpp"
#include "state.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>

#define CUDA_STACK_SIZE 1024 * 32

#define SEED 12345

//#define NUM_BLOCKS 8 //192
#define BLOCK_SIZE 128
#define UNDERFILL 4

__global__ void coarsePlayoutKernel(State *states, PlayerId *results, size_t numStates, uint32_t *globalStateIndex) {
  uint8_t tx = threadIdx.x;
  uint32_t bx = blockIdx.x;
  uint32_t tid = tx + (bx * blockDim.x);

  uint32_t threadStateIndex = tid;

  if (tid < numStates) {
    State state;
    state = states[threadStateIndex];

    // Init random generator
    curandState_t generator;
    curand_init(SEED, threadStateIndex, 0, &generator);

    Move captureMoves[MAX_MOVES];
    Move directMoves[MAX_MOVES];
    uint8_t numMoveCapture, numMoveDirect;

    while (1) {
      bool gameOver = false;
      // Check if the game is a draw
      if (state.movesSinceLastCapture >= NUM_DRAW_MOVES) {
        results[tid] = PLAYER_NONE;
        gameOver = true;
      } else {
        // Scan the board for pieces that can move
        numMoveCapture = 0;
        numMoveDirect = 0;
        for (uint8_t i = 0; i < BOARD_SIZE; i++) {
          for (uint8_t j = 1 - (i % 2); j < BOARD_SIZE; j+=2) {
            Loc here(i, j);
            numMoveCapture += state.genLocMoves(here, &captureMoves[numMoveCapture], SINGLE_CAPTURE);
            numMoveDirect += state.genLocMoves(here, &directMoves[numMoveDirect], DIRECT);
          }
        }
      
        // Perform a random move if there is one
        if (numMoveCapture > 0) {
          do {
            uint8_t moveIndex = curand(&generator) % numMoveCapture;
            Loc to = captureMoves[moveIndex].to;
            state.move(captureMoves[moveIndex]);
            state.turn = state.getNextTurn();
            numMoveCapture = state.genLocMoves(to, captureMoves, SINGLE_CAPTURE);
          } while (numMoveCapture > 0);
          state.turn = state.getNextTurn();
        } else if (numMoveDirect > 0) {
          state.move(directMoves[curand(&generator) % numMoveDirect]);
        } else {
          // If the game is over, write the winner to the results array
          results[threadStateIndex] = state.getNextTurn();
          gameOver = true;
        }
      }
      if (gameOver) {
        // Attempt to select another state from the inputs
        if (*globalStateIndex >= numStates) {
          break;
        } else {
          unsigned oldGlobalStateIndex;
          do {
            threadStateIndex = *globalStateIndex;
            oldGlobalStateIndex = atomicCAS(globalStateIndex, threadStateIndex, threadStateIndex + 1);
          } while (oldGlobalStateIndex != threadStateIndex);
          state = states[threadStateIndex]; 
        }
        gameOver = false;
      }
    }
  }
}

std::vector<PlayerId> DeviceCoarsePlayoutDriver::runPlayouts(std::vector<State> states) {
  // Query device to figure out how many blocks can run in parallel
  int device;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error getting device: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
  struct cudaDeviceProp properties;
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error getting device properties: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  unsigned numBlocks =
    properties.multiProcessorCount * properties.maxThreadsPerMultiProcessor / (BLOCK_SIZE * UNDERFILL);
  std::cout << "Launching " << numBlocks << " blocks" << std::endl;
  
  // Device variables
  State *devStates;
  PlayerId *devResults;
  cudaMalloc(&devStates, states.size() * sizeof(State));
  cudaMalloc(&devResults, states.size() * sizeof(PlayerId));
 
  uint32_t *globalStateIndex;
  cudaMalloc((void**) &globalStateIndex, sizeof(uint32_t));

  // Copy states for playouts to device
  cudaMemcpy(devStates, states.data(), states.size() * sizeof(State), cudaMemcpyHostToDevice);
  
  // Copy global state index to be the number of threads initially
  unsigned numThreads = numBlocks * BLOCK_SIZE; // max number of threads that can run in parallel
  cudaMemcpy(globalStateIndex, &numThreads, sizeof(uint32_t), cudaMemcpyHostToDevice);

  // Increase default stack size
  error = cudaDeviceSetLimit(cudaLimitStackSize, CUDA_STACK_SIZE);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error setting stack size: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  // Invoke the kernel
  coarsePlayoutKernel<<<numBlocks, BLOCK_SIZE>>>(devStates, devResults, states.size(), globalStateIndex);
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
