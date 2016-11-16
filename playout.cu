
#include "state.hpp"
#include "playout.hpp"

__global__ void playoutKernel(State *states, PlayerId *results) {
  uint8_t id = threadIdx.x;
  uint8_t row = id / (BOARD_SIZE / 2);
  uint8_t col = id % (BOARD_SIZE / 2) + (row % 2 == 0);
  Loc loc(row, col);

  __shared__ State state;
  state = states[blockIdx.x];

  __shared__ Move directMoves[NUM_PLAYERS][MAX_MOVES];
  __shared__ Move captureMoves[NUM_PLAYERS][MAX_MOVES];
  __shared__ uint8_t directMoveIndices[NUM_PLAYERS][NUM_LOCS];
  __shared__ uint8_t captureMoveIndices[NUM_PLAYERS][NUM_LOCS];

  while (!state.isFinished()) {
    PlayerId locOwner = state[loc].owner;

    // Generate direct and capture moves for this location
    Move locDirectMoves[MAX_LOC_MOVES];
    Move locCaptureMoves[MAX_LOC_MOVES];
    uint8_t numLocDirectMoves  = state.locDirectMoves(loc, locDirectMoves);
    uint8_t numLocCaptureMoves = state.locCaptureMoves(loc, locCaptureMoves);

    for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
      if (locOwner == (PlayerId)i) {
	directMoveIndices[id][i]  = numLocDirectMoves;
	captureMoveIndices[id][i] = numLocCaptureMoves;
      }
      else {
	directMoveIndices[id][i]  = 0;
	captureMoveIndices[id][i] = 0;
      }
    }

    // Perform a reduction to calculate the max number of captures possible for each player
    for (unsigned i = NUM_LOCS / 2; i > 0; i >>= 1) {
      __syncthreads();
      if (i > id) {
	for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	  if (captureMoveIndices[i][id + i] > captureMoveIndices[i][id])
	    captureMoveIndices[i][id] = captureMoveIndices[i][id + i];
	}
      }
    }

    __shared__ uint8_t maxCaptureMoves[NUM_PLAYERS];
    if (id < NUM_PLAYERS)
      maxCaptureMoves[id] = captureMoveIndices[id][0];
    __syncthreads();

    // The number of capture moves for a location is 0 if there are any other locations with more capture moves
    for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
      captureMoveIndices[i][id] = 0;
    }
    if (numLocCaptureMoves < maxCaptureMoves[locOwner])
      numLocCaptureMoves = 0;
    captureMoveIndices[locOwner][id] = numLocCaptureMoves;

    // Perform exclusive scans to get indices to copy moves into the shared arrays
    for (uint8_t stride = 1; stride <= NUM_LOCS; stride <<= 1) {
      __syncthreads();
      uint8_t i = (id + 1) * stride - 1; // TODO: Check that this is correct...
      if (i < NUM_LOCS) {
	for (uint8_t j = 0; j < NUM_PLAYERS; j++) {
	  directMoveIndices[j][i]  += directMoveIndices[j][i - stride];
	  captureMoveIndices[j][i] += captureMoveIndices[j][i - stride];
	}
      }
    }

    __syncthreads();
    if (id < NUM_PLAYERS) {
      directMoveIndices[id][NUM_LOCS - 1] = 0;
      captureMoveIndices[id][NUM_LOCS - 1] = 0;
    }

    for (uint8_t stride = NUM_LOCS / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      int i = (id + 1) * stride - 1;
      uint8_t temp;
      for (uint8_t j = 0; j < NUM_PLAYERS; j++) {
	temp = directMoveIndices[j][i];
	directMoveIndices[j][i] += directMoveIndices[j][i - stride];
	directMoveIndices[j][i - stride] = temp;
	temp = captureMoveIndices[j][i];
	captureMoveIndices[j][i] += captureMoveIndices[j][i - stride];
	captureMoveIndices[j][i - stride] = temp;
      }
    }

    __syncthreads();

    // Copy generated moves to shared arrays
    for (uint8_t i = 0; i < numLocDirectMoves; i++) {
      //directMoves[i + directMoveIndices[id]] = locDirectMoves[i];
    }
  }
}

std::vector<PlayerId> playouts(std::vector<State>) {
  
}
