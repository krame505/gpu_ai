
#include "playout.hpp"
#include "state.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <vector>

#define CUDA_STACK_SIZE 1024 * 32

#define SEED 12345

#define BLOCK_SIZE 32

#define USE_SIMPLE_CAPTURE

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

    Move captureMoves[MAX_LOC_MOVES * 12];
    Move directMoves[MAX_LOC_MOVES * 12];
    uint8_t numMoveCapture, numMoveDirect;

    do {
      // Scan the board for pieces that can move
      numMoveCapture = 0;
      numMoveDirect = 0;
      for (uint8_t i = 0; i < BOARD_SIZE; i++) {
	for (uint8_t j = 1 - (i % 2); j < BOARD_SIZE; j+=2) {
	  Loc here(i, j);
#ifdef USE_SIMPLE_CAPTURE
	  numMoveCapture += state.genLocCaptureMovesSimple(here, &captureMoves[numMoveCapture]);
#else
	  numMoveCapture += state.genLocCaptureMoves(here, &captureMoves[numMoveCapture]);
#endif
	  numMoveDirect += state.genLocDirectMoves(here, &directMoves[numMoveDirect]);
	}
      }

      if (numMoveCapture > 0) {
#ifdef USE_SIMPLE_CAPTURE
	do {
	  uint8_t moveIndex = curand(&generator) % numMoveCapture;
	  Loc to = captureMoves[moveIndex].to;
	  state.move(captureMoves[moveIndex]);
	  state.turn = state.getNextTurn();
	  numMoveCapture = state.genLocCaptureMovesSimple(to, captureMoves);
	} while (numMoveCapture > 0);
	state.turn = state.getNextTurn();
#else
	state.move(captureMoves[curand(&generator) % numMoveCapture]);
#endif
      }
      else if (numMoveDirect > 0) {
	state.move(directMoves[curand(&generator) % numMoveDirect]);
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

  int numBlocks = states.size() / BLOCK_SIZE;
  if (states.size() % BLOCK_SIZE)
    numBlocks++;

  cudaError_t error;

#ifndef USE_SIMPLE_CAPTURE
  error = cudaDeviceSetLimit(cudaLimitStackSize, CUDA_STACK_SIZE);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error setting stack size: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
#endif

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
