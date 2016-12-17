
#include "playout.hpp"
#include "state.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>

#define CUDA_STACK_SIZE 1024 * 32

#define SEED 12345

__global__ void playoutKernel(State *states, PlayerId *results) {
  uint8_t tx = threadIdx.x;
  uint32_t bx = blockIdx.x;
  uint32_t tid = tx + (bx * NUM_LOCS);

  __shared__ State state;
  state = states[bx];

  // Init random generator
  curandState_t generator;
  curand_init(SEED, tid, 0, &generator);
 
  // __shared__ uint8_t numDirectMoves[NUM_PLAYERS];
  // __shared__ uint8_t numCaptureMoves[NUM_PLAYERS];
  // __shared__ Move directMoves[NUM_PLAYERS][MAX_MOVES];
  // __shared__ Move captureMoves[NUM_PLAYERS][MAX_MOVES];
 
  __shared__ Move moves[MAX_MOVES];

  __shared__ bool gameOver;

  if (tx == 0)
    gameOver = false;

  do {
    uint8_t numMoves = state.genMoves(moves);

    if (tx == 0) {
      // Select a move
      Move move;
      if (numMoves > 0) {
        move = moves[curand(&generator) % numMoves];
      }
      else {
        gameOver = true; // No moves, game is over
      }

      // Perform the move if there is one
      if (!gameOver)
	state.move(move);
    }
  } while (!gameOver);

  if (tx == 0)
    results[bx] = state.getNextTurn();
}

std::vector<PlayerId> DevicePlayoutDriver::runPlayouts(std::vector<State> states) const {
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
  playoutKernel<<<states.size(), NUM_LOCS>>>(devStates, devResults);
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
