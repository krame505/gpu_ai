
#include "playout.hpp"
#include "player.hpp"

#include <vector>
#include <cassert>
using namespace std;

#define NUM_THREADS 8
vector<PlayerId> hostPlayouts(vector<State> states) {
  vector<PlayerId> results(states.size());
  RandomPlayer player;

  #pragma omp parallel for
  for (unsigned i = 0; i < states.size(); i++) {
    State state = states[i];
    while (!state.isFinished()) {
      Move move = player.getMove(state);
      state.move(move);
    }
    results[i] = state.result();
  }
  
  return results;
}
