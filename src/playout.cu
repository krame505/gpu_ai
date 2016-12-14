
#include "playout.hpp"
#include "state.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>

#define SEED 12345

__global__ void playoutKernel(State *states, PlayerId *results) {
  INIT_KERNEL_VARS

  __shared__ State state;
  state = states[blockIdx.x];

  // Init random generators
  __shared__ curandState_t generators[NUM_LOCS];
  curand_init(SEED, id, 1, &generators[id]);
 
  // __shared__ uint8_t numDirectMoves[NUM_PLAYERS];
  // __shared__ uint8_t numCaptureMoves[NUM_PLAYERS];
  // __shared__ Move directMoves[NUM_PLAYERS][MAX_MOVES];
  // __shared__ Move captureMoves[NUM_PLAYERS][MAX_MOVES];
 
  __shared__ uint8_t numMoves[NUM_PLAYERS];
  __shared__ Move moves[NUM_PLAYERS][MAX_MOVES];

  __shared__ bool gameOver;

  if (threadIdx.x == 0)
    gameOver = false;

  do {
    state.genMoves(numMoves, moves);

    // Select a move
    // TODO: Optimize this portion
    if (id == 0) {
      Move move;
      if (numMoves[state.turn] > 0) {
        move = moves[state.turn][curand(&generators[id]) % numMoves[state.turn]];
      }
      else {
        gameOver = true; // No moves, game is over
      }

      // Perform the move if there is one
      if (!gameOver)
	state.move(move);
    }
  } while (!gameOver);

  // TODO: Implement State::result to make use of parallelism
  if (threadIdx.x == 0)
    results[id] = state.result();
}

std::vector<PlayerId> DevicePlayoutDriver::runPlayouts(std::vector<State> states) const {
  // Device variables
  State *devStates;
  PlayerId *devResults;
  cudaMalloc(&devStates, states.size() * sizeof(State));
  cudaMalloc(&devResults, states.size() * sizeof(PlayerId));

  // Copy states for playouts to device
  cudaMemcpy(devStates, states.data(), states.size() * sizeof(State), cudaMemcpyHostToDevice);

  // Invoke the kernel
  playoutKernel<<<NUM_LOCS, states.size()>>>(devStates, devResults);
  cudaDeviceSynchronize();

  // Check for errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  // Copy the results back to the host
  PlayerId results[states.size()];
  cudaMemcpy(results, devResults, states.size() * sizeof(PlayerId), cudaMemcpyDeviceToHost);

  // Return a vector constructed from the contents of the array
  return std::vector<PlayerId>(results, results + states.size());
}
