
#include "playout.hpp"
#include "player.hpp"

#include <vector>
#include <cassert>
#include <future>
using namespace std;

#define NUM_THREADS 8

vector<PlayerId> worker(const vector<State> &states) {
  RandomPlayer player;
  vector<PlayerId> results;
  for (State state : states) {
    while (!state.isFinished()) {
      Move move = player.getMove(state);
      state.move(move);
    }
    results.push_back(state.result());
  }
  return results;
}

vector<PlayerId> hostPlayouts(vector<State> states) {
  unsigned blockSize = states.size() / NUM_THREADS;
  vector<future<vector<PlayerId>>> results;

  for (auto it = states.begin(); it < states.end(); it += blockSize) {
    vector<State> threadStates;
    if (it + blockSize < states.end())
      threadStates = vector<State>(it, it + blockSize);
    else
      threadStates = vector<State>(it, states.end());
    results.push_back(async(launch::async, worker, threadStates));
  }
 
  vector<PlayerId> result;
  for (future<vector<PlayerId>> &res : results) {
    vector<PlayerId> threadResult = res.get();
    result.insert(result.end(), threadResult.begin(), threadResult.end());
  }
  
  return result;
}
