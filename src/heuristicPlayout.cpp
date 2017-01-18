
#include "playout.hpp"
#include "heuristic.hpp"

#include <omp.h>

#include <vector>
#include <random>
#include <cassert>
using namespace std;

vector<PlayerId> HostHeuristicPlayoutDriver::runPlayouts(vector<State> states) {
  vector<PlayerId> results(states.size());

  default_random_engine generator;
  normal_distribution<float> distribution(0, HEURISTIC_SIGMA);

  #pragma omp parallel for
  for (unsigned i = 0; i < states.size(); i++) {
    State state = states[i];
    unsigned stateScore[NUM_PLAYERS];
    scoreState(state, stateScore);
    while (!state.isGameOver()) {
      vector<Move> moves = state.getMoves();
      Move optMove;
      int optMoveScore[NUM_PLAYERS] = {0, 0};
      float optWeight = -1/0.0; // -infinity
      for (Move move : moves) {
	int moveScore[NUM_PLAYERS];
	scoreMove(state, move, moveScore);
	float weight = getWeight(state, stateScore, moveScore) + distribution(generator);
	if (weight > optWeight) {
	  optMove = move;
	  optWeight = weight;
	  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	    optMoveScore[i] = moveScore[i];
	  }
	}
      }
      state.move(optMove);
      for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
	stateScore[i] += optMoveScore[i];
      }
    }
    results[i] = state.getNextTurn();
  }
  
  return results;
}
