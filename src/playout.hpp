#pragma once

#include "state.hpp"

#include <vector>
#include <string>

// In theory should be host runtime / device runtime for target # of playouts
// But in practice needs tuning for MCTS
#define INITIAL_DEVICE_HOST_PLAYOUT_RATIO 1.35
#define DEVICE_HOST_PLAYOUT_RATIO_SCALE 1.05

#define HOST_MAX_PLAYOUT_SIZE 300
#define HYBRID_MAX_PLAYOUT 4000

class PlayoutDriver {
public:
  virtual ~PlayoutDriver() {}
  
  virtual std::vector<PlayerId> runPlayouts(std::vector<State>) = 0;
  virtual std::string getName() const = 0;
};

// Perform playouts on the CPU from the provided states, returning the winners
class HostPlayoutDriver : public PlayoutDriver {
public:
  ~HostPlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "host"; }
};

// Perform playouts on the CPU using a heuristic for determining the winner
// from the provided states, returning the winners
class HostHeuristicPlayoutDriver : public PlayoutDriver {
public:
  ~HostHeuristicPlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "host_heuristic"; }
};

// Perform playouts on the GPU with 1 state per thread from the provided states,
// returning the winners
class DeviceSinglePlayoutDriver : public PlayoutDriver {
public:
  ~DeviceSinglePlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "device_single"; }
};

// Perform playouts on the GPU with 1 state per thread from the provided states,
// returning the winners
class DeviceCoarsePlayoutDriver : public PlayoutDriver {
public:
  ~DeviceCoarsePlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "device_coarse";}
};

// Perform playouts on the GPU with 1 state per block using a heuristic for determining the winner
// from the provided states, returning the winners
class DeviceHeuristicPlayoutDriver : public PlayoutDriver {
public:
  ~DeviceHeuristicPlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "device_heuristic"; }
};

// Perform playouts on the GPU with 1 state per block from the provided states,
// returning the winners
class DeviceMultiplePlayoutDriver : public PlayoutDriver {
public:
  ~DeviceMultiplePlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "device_multiple"; }
};

// Perform playouts on the GPU and CPU in parallel from the provided states,
// returning the winners
class HybridPlayoutDriver : public PlayoutDriver {
public:
  HybridPlayoutDriver(float deviceHostPlayoutRatio=INITIAL_DEVICE_HOST_PLAYOUT_RATIO) :
    deviceHostPlayoutRatio(deviceHostPlayoutRatio) {}
  ~HybridPlayoutDriver() {}

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "hybrid"; }

private:
  float deviceHostPlayoutRatio;
};

// Perform playouts using whichever of the above is appropriate for the provided number of playouts
class OptimalPlayoutDriver : public PlayoutDriver {
public:
  OptimalPlayoutDriver() {}
  ~OptimalPlayoutDriver() {}

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "optimal"; }
};

PlayoutDriver *getPlayoutDriver(std::string name);
