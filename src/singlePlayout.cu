
#include "playout.hpp"
#include "state.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>

#define CUDA_STACK_SIZE 1024 * 32

#define SEED 12345

#define BLOCK_SIZE 128

__global__ void singlePlayoutKernel(State *states, PlayerId *results, size_t numStates) {
  uint32_t tx = threadIdx.x;
  uint32_t bx = blockIdx.x;
  uint32_t tid = tx + (bx * BLOCK_SIZE);

  if (tid < numStates) {
    State state;
    state = states[tid];

    // Init random generator
    curandState_t generator;
    curand_init(SEED, tid, 0, &generator);

    Move captureMoves[MAX_MOVES];
    Move directMoves[MAX_MOVES];
    uint8_t numMoveCapture, numMoveDirect;

    while (1) {
      // Check if the game is a draw
      if (state.movesSinceLastCapture >= NUM_DRAW_MOVES) {
        results[tid] = PLAYER_NONE;
        break;
      }
    
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
        results[tid] = state.getNextTurn();
        break;
      }
    }
  }
}

std::vector<PlayerId> DeviceSinglePlayoutDriver::runPlayouts(std::vector<State> states) {
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

  // Increase default stack size
  cudaError_t error = cudaDeviceSetLimit(cudaLimitStackSize, CUDA_STACK_SIZE);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error setting stack size: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  // Invoke the kernel
  singlePlayoutKernel<<<numBlocks, BLOCK_SIZE>>>(devStates, devResults, states.size());
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
