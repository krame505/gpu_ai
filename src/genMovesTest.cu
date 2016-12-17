
#include "genMovesTest.hpp"
#include "state.hpp"

#define CUDA_STACK_SIZE 1024 * 32

#include <iostream>
using namespace std;

__global__ void genMovesKernel(State *globalState, Move *globalMoves, uint8_t *globalNumMoves) {
  __shared__ State state;
  __shared__ Move moves[MAX_MOVES];

  state = *globalState;
  uint8_t numMoves = state.genMoves(moves);

  if (threadIdx.x == 0)
    *globalNumMoves = numMoves;

  for (unsigned i = 0; i < numMoves; i++) {
    unsigned index = i + threadIdx.x;
    globalMoves[index] = moves[index];
  }
}

bool genMovesTest(State state) {
  // Device variables
  State *devState;
  Move *devMoves;
  uint8_t *devNumMoves;
  cudaMalloc(&devState, sizeof(State));
  cudaMalloc(&devMoves, MAX_MOVES * sizeof(Move));
  cudaMalloc(&devNumMoves, sizeof(uint8_t));

  // Copy states for playouts to device
  cudaMemcpy(devState, &state, sizeof(State), cudaMemcpyHostToDevice);

  cudaError_t error = cudaDeviceSetLimit(cudaLimitStackSize, CUDA_STACK_SIZE);
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error setting stack size: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  // Invoke the kernel
  genMovesKernel<<<1, NUM_LOCS>>>(devState, devMoves, devNumMoves);
  cudaDeviceSynchronize();

  // Check for errors
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    exit(1);
  }

  // Copy the results back to the host
  Move gpuMoves[MAX_MOVES];
  cudaMemcpy(gpuMoves, devMoves, MAX_MOVES * sizeof(Move), cudaMemcpyDeviceToHost);
  
  uint8_t gpuNumMoves;
  cudaMemcpy(&gpuNumMoves, devNumMoves, sizeof(uint8_t), cudaMemcpyDeviceToHost);

  // Free the global memory
  cudaFree(devState);
  cudaFree(devMoves);
  cudaFree(devNumMoves);

  Move cpuMoves[MAX_MOVES];
  uint8_t cpuNumMoves = state.genMoves(cpuMoves);

  bool match = true;

  if (cpuNumMoves != gpuNumMoves)
    match = false;

  for (uint8_t i = 0; i < cpuNumMoves; i++) {
    if (cpuMoves[i] != gpuMoves[i]) {
      match = false;
      break;
    }
  }

  if (!match) {
    cout << "Mismatch in CPU and GPU genMoves()" << endl;
    cout << state << endl;

    cout << "CPU Moves: " << (int)cpuNumMoves << endl;
    for (uint8_t i = 0; i < cpuNumMoves; i++) {
      cout << cpuMoves[i] << endl;
    }

    cout << "GPU Moves: " << (int)gpuNumMoves << endl;
    for (uint8_t i = 0; i < gpuNumMoves; i++) {
      cout << gpuMoves[i] << endl;
    }
  }

  return match;
}
