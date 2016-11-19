
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

  // TODO: Figure out who won
}

std::vector<PlayerId> playouts(std::vector<State> states) {
  PlayerId results[states.size()];
  //playoutKernel<<NUM_LOCS, states.size()>>>(states.begin(), results); // TODO: Why doesn't this line compile?  
  return std::vector<PlayerId>(results, results + states.size());
}
