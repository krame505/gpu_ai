
#include "playout.hpp"
#include "player.hpp"

#include <omp.h>

#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <climits>
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
    results[i] = state.getWinner();
  }
  
  return results;
}

vector<PlayerId> HybridPlayoutDriver::runPlayouts(vector<State> states) {
  // Calculate # of playouts to perform on GPU and CPU
  unsigned numDeviceTrials = (states.size() * deviceHostPlayoutRatio) / (1 + deviceHostPlayoutRatio);
  unsigned numHostTrials = states.size() / (1 + deviceHostPlayoutRatio);
  if (numHostTrials + numDeviceTrials < states.size())
    numDeviceTrials++;

  DeviceMultiplePlayoutDriver devicePlayoutDriver;
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

vector<PlayerId> OptimalPlayoutDriver::runPlayouts(vector<State> states) {
  // Choose a playout driver
  unsigned trials = states.size();
  //cout << "Allocating " << trials << " trials" << endl;
  double totalScore = 0;
  vector<double> scores(playoutDrivers.size());
  vector<double> confidences(playoutDrivers.size());
  for (unsigned i = 0; i < playoutDrivers.size(); i++) {
    auto result = score(trials, prevRuntimes[i]);
    totalScore += result.first;
    scores[i] = result.first;
    confidences[i] = result.second;
  }
  vector<double> weights(playoutDrivers.size());
  for (unsigned i = 0; i < playoutDrivers.size(); i++) {
    weights[i] = (scores[i] == INFINITY? 0 : scores[i] / totalScore) + confidences[i];
    //cout << i << ": " << weights[i] << endl;
  }
  discrete_distribution<> d(weights.begin(), weights.end());
  unsigned driverIndex = d(gen);
  auto &chosenDriver = playoutDrivers[driverIndex];
  auto &chosenPrevRuntimes = prevRuntimes[driverIndex];
  //cout << "Picked " << driverIndex << ": " << chosenDriver->getName() << endl;

  // Perform the playouts and measure the runtime
  chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
  vector<PlayerId> results = chosenDriver->runPlayouts(states);
  chrono::high_resolution_clock::time_point finish = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsedTime = finish - start;

  // Update the measured runtimes for that driver, deleting outdated entries
  for (auto it = chosenPrevRuntimes.lower_bound(trials);
       it != chosenPrevRuntimes.end() && it->second < elapsedTime;
       it = chosenPrevRuntimes.erase(it));
  for (auto it =
         map<unsigned, chrono::duration<double>>::reverse_iterator(chosenPrevRuntimes.lower_bound(trials));
       it != chosenPrevRuntimes.rend() && it->second > elapsedTime;
       it = map<unsigned, chrono::duration<double>>::reverse_iterator(chosenPrevRuntimes.erase(next(it).base())));
  chosenPrevRuntimes[trials] = elapsedTime;

  return results;
}

pair<double, double> OptimalPlayoutDriver::score(const unsigned trials,
                                                 const map<unsigned, chrono::duration<double>> &prevRuntimes) {
  chrono::duration<double> predictedRuntime;
  double confidence;
  if (prevRuntimes.count(trials)) {
    predictedRuntime = prevRuntimes.at(trials);
    confidence = 0;
  }
  else if (prevRuntimes.size() < 2) {
    predictedRuntime = chrono::duration<double>(0);
    confidence = INFINITY;
  } else {
    // Find the 2 elements of prevRuntimes that are closest to trials
    auto it = prevRuntimes.upper_bound(trials);
    for (unsigned i = 0; i < 2 && it != prevRuntimes.begin(); i++, it--);
    unsigned trials1 = 0;
    chrono::duration<double> time1;
    unsigned minDifference1 = UINT_MAX;
    unsigned trials2 = 0;
    chrono::duration<double> time2;
    unsigned minDifference2 = UINT_MAX;
    for (unsigned i = 0; i < 4 && it != prevRuntimes.end(); i++, it++) {
      unsigned difference = it->first > trials? it->first - trials : trials - it->first;
      if (difference < minDifference1) {
        trials2 = trials1;
        time2 = time1;
        minDifference2 = minDifference1;
        trials1 = it->first;
        time1 = it->second;
        minDifference1 = difference;
      } else if (difference < minDifference2) {
        trials2 = it->first;
        time2 = it->second;
        minDifference2 = difference;
      }
    }
    double slope = (time1 - time2).count() / (signed)(trials1 - trials2);
    predictedRuntime = time1 + chrono::duration<double>(slope * (signed)(trials - trials1));
    confidence = pow(minDifference1 * minDifference2, OPTIMAL_CONFIDENCE_EXP) * OPTIMAL_CONFIDENCE_SCALE / trials;
  }

  double score = 1 / pow(predictedRuntime.count(), OPTIMAL_SCORE_EXP);
  return pair<double, double>(score, confidence);
}

vector<unique_ptr<PlayoutDriver>> OptimalPlayoutDriver::getDefaultPlayoutDrivers() {
  vector<unique_ptr<PlayoutDriver>> playoutDrivers;
  playoutDrivers.push_back(make_unique<HostPlayoutDriver>());
  //playoutDrivers.push_back(make_unique<HybridPlayoutDriver>(make_unique<HostPlayoutDriver>(),
  //                                                          make_unique<DeviceMultiplePlayoutDriver>()));
  playoutDrivers.push_back(make_unique<DeviceMultiplePlayoutDriver>());
  playoutDrivers.push_back(make_unique<HybridPlayoutDriver>(make_unique<HostPlayoutDriver>(),
                                                            make_unique<DeviceCoarsePlayoutDriver>()));
  return playoutDrivers;
}

vector<PlayerId> OptimalHeuristicPlayoutDriver::runPlayouts(vector<State> states) {
  HostHeuristicPlayoutDriver hostPlayoutDriver;
  HybridHeuristicPlayoutDriver hybridPlayoutDriver;
  
  if (states.size() < HOST_MAX_PLAYOUT_SIZE)
    return hostPlayoutDriver.runPlayouts(states);
  else
    return hybridPlayoutDriver.runPlayouts(states);
}

unique_ptr<PlayoutDriver> getPlayoutDriver(string name) {
  if (name == "host") {
    return make_unique<HostPlayoutDriver>();
  }
  else if (name == "host_heuristic") {
    return make_unique<HostHeuristicPlayoutDriver>();
  }
  else if (name == "device_single") {
    return make_unique<DeviceSinglePlayoutDriver>();
  }
  else if (name == "device_multiple") {
    return make_unique<DeviceMultiplePlayoutDriver>();
  }
  else if (name == "device_coarse") {
    return make_unique<DeviceCoarsePlayoutDriver>();
  }
  else if (name == "device_heuristic") {
    return make_unique<DeviceHeuristicPlayoutDriver>();
  }
  else if (name == "hybrid") {
    return make_unique<HybridPlayoutDriver>();
  }
  else if (name == "hybrid_heuristic") {
    return make_unique<HybridPlayoutDriver>();
  }
  else if (name == "optimal") {
    return make_unique<OptimalPlayoutDriver>();
  }
  else if (name == "optimal_heuristic") {
    return make_unique<OptimalHeuristicPlayoutDriver>();
  }
  else {
    throw runtime_error("Unknown playout type");
  }
}
