
#include "playout.hpp"
#include "heuristic.hpp"
#include "player.hpp"

#include <omp.h>

#include <vector>
#include <cassert>
using namespace std;

vector<PlayerId> HostHeuristicPlayoutDriver::runPlayouts(vector<State> states) {
  vector<PlayerId> results(states.size());
  RandomPlayer player;

  #pragma omp parallel for
  for (unsigned i = 0; i < states.size(); i++) {
    State state = states[i];
    
    PlayerId predictedWinner = predictWinner(state);
    while (predictedWinner == PLAYER_NONE) {
      if (state.isGameOver()) {
	predictedWinner = state.getNextTurn();
      }
      else {
	Move move = player.getMove(state);
	state.move(move);
	predictedWinner = predictWinner(state);
      }
    }
    results[i] = predictedWinner;
  }
  
  return results;
}
