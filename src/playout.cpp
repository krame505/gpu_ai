
#include "playout.hpp"
#include "player.hpp"

#include <vector>
#include <random>
#include <cassert>
using namespace std;

vector<PlayerId> HostPlayoutDriver::runPlayouts(vector<State> states) const {
  vector<PlayerId> results(states.size());
  RandomPlayer player;

  #pragma omp parallel for
  for (unsigned i = 0; i < states.size(); i++) {
    State state = states[i];
    while (!state.isGameOver()) {
      Move move = player.getMove(state);
      state.move(move);
    }
    results[i] = state.getNextTurn();
  }
  
  return results;
}
