
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
    while (!state.isGameOver()) {
      vector<Move> moves = state.getMoves();
      Move optMove;
      float optScore = -1/0.0; // -infinity
      for (Move move : moves) {
	State newState = state;
	newState.move(move);
	float score = scoreHeuristic(newState) + distribution(generator);
	if (score > optScore) {
	  optMove = move;
	  optScore = score;
	}
      }
      state.move(optMove);
    }
    results[i] = state.getNextTurn();
  }
  
  return results;
}
