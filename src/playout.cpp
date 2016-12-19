
#include "playout.hpp"
#include "player.hpp"

#include <omp.h>

#include <vector>
#include <random>
#include <cassert>
#include <chrono>
using namespace std;

vector<PlayerId> HostPlayoutDriver::runPlayouts(vector<State> states) {
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

vector<PlayerId> HybridPlayoutDriver::runPlayouts(vector<State> states) {
  // Calculate # of playouts to perform on GPU and CPU
  unsigned numDeviceTrials = (states.size() * deviceHostPlayoutRatio) / (1 + deviceHostPlayoutRatio);
  unsigned numHostTrials = states.size() / (1 + deviceHostPlayoutRatio);
  if (numHostTrials + numDeviceTrials < states.size())
    numDeviceTrials++;

  DeviceSinglePlayoutDriver devicePlayoutDriver;
  HostPlayoutDriver hostPlayoutDriver;

  vector<PlayerId> results(numDeviceTrials);
  vector<PlayerId> hostResults(numHostTrials);

  chrono::high_resolution_clock::time_point deviceFinished, hostFinished;

  // Run playouts in parallel on CPU and GPU
  omp_set_nested(1);
#pragma omp parallel num_threads(2)
  {
    if (omp_get_thread_num() == 0) {
      vector<State> deviceStates(states.begin(), states.begin() + numDeviceTrials);
      results = devicePlayoutDriver.runPlayouts(deviceStates);
      deviceFinished = chrono::high_resolution_clock::now();
    }
    else {
      vector<State> hostStates(states.begin() + numDeviceTrials, states.end());
      hostResults = hostPlayoutDriver.runPlayouts(hostStates);
      hostFinished = chrono::high_resolution_clock::now();
    }
  }
  omp_set_nested(0);

  // Assemble host and device results into result array
  results.insert(results.end(), hostResults.begin(), hostResults.end());

  // Update ratio between # of host and device playouts
  chrono::duration<double> diff = deviceFinished - hostFinished;
  if (diff.count() < 0)
    deviceHostPlayoutRatio *= DEVICE_HOST_PLAYOUT_RATIO_SCALE;
  else
    deviceHostPlayoutRatio /= DEVICE_HOST_PLAYOUT_RATIO_SCALE;
  //cout << "Ratio: " << deviceHostPlayoutRatio << endl;

  return results;
}

PlayoutDriver *getPlayoutDriver(string name) {
  if (name == "host") {
    return new HostPlayoutDriver;
  }
  else if (name == "device_single") {
    return new DeviceSinglePlayoutDriver;
  }
  else if (name == "device_heuristic") {
    return new DeviceHeuristicPlayoutDriver;
  }
  else if (name == "device_multiple") {
    return new DeviceMultiplePlayoutDriver;
  }
  else if (name == "device_relaunch") {
    return new DeviceRelaunchPlayoutDriver;
  }
  else if (name == "hybrid") {
    return new HybridPlayoutDriver;
  }
  else {
    throw runtime_error("Unknown playout type");
  }
}
