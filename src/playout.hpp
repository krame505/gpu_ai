#pragma once

#include "state.hpp"

#include <vector>
#include <string>

// In theory should be host runtime / device runtime for target # of playouts
// But in practice needs tuning for MCTS
#define INITIAL_HYBRID_PLAYOUT_RATIO 1.35
#define INITIAL_HYBRID_HEURISTIC_PLAYOUT_RATIO 0.9
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

// Perform playouts on the CPU using a heuristic for selecting moves
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

// Perform playouts on the GPU with 1 state per block using a heuristic for selecting moves
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
  HybridPlayoutDriver(PlayoutDriver *hostPlayoutDriver=new DeviceMultiplePlayoutDriver,
		      PlayoutDriver *devicePlayoutDriver=new HostPlayoutDriver,
		      float deviceHostPlayoutRatio=INITIAL_HYBRID_PLAYOUT_RATIO) :
    hostPlayoutDriver(hostPlayoutDriver),
    devicePlayoutDriver(devicePlayoutDriver),
    deviceHostPlayoutRatio(deviceHostPlayoutRatio) {}
  HybridPlayoutDriver(float deviceHostPlayoutRatio) :
    HybridPlayoutDriver(new DeviceMultiplePlayoutDriver,
			new HostPlayoutDriver,
			deviceHostPlayoutRatio) {}
  virtual ~HybridPlayoutDriver() {
    delete hostPlayoutDriver;
    delete devicePlayoutDriver;
  }

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  virtual std::string getName() const { return "hybrid"; }

private:
  PlayoutDriver *hostPlayoutDriver;
  PlayoutDriver *devicePlayoutDriver;
  float deviceHostPlayoutRatio;
};

// Perform playouts on the GPU and CPU in parallel using a heuristic for selecting moves
// from the provided states, returning the winners
class HybridHeuristicPlayoutDriver : public HybridPlayoutDriver {
public:
  HybridHeuristicPlayoutDriver(float deviceHostPlayoutRatio=INITIAL_HYBRID_HEURISTIC_PLAYOUT_RATIO) :
    HybridPlayoutDriver(new HostHeuristicPlayoutDriver,
				 new DeviceHeuristicPlayoutDriver,
				 deviceHostPlayoutRatio) {}

  std::string getName() const { return "hybrid_heuristic"; }
};

// Perform playouts using whichever of the above is appropriate for the provided number of playouts
class OptimalPlayoutDriver : public PlayoutDriver {
public:
  OptimalPlayoutDriver() {}
  ~OptimalPlayoutDriver() {}

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "optimal"; }
};

// Perform heuristic playouts using whichever of the above is appropriate for the provided number of playouts
class OptimalHeuristicPlayoutDriver : public PlayoutDriver {
public:
  OptimalHeuristicPlayoutDriver() {}
  ~OptimalHeuristicPlayoutDriver() {}

  std::vector<PlayerId> runPlayouts(std::vector<State>);
  std::string getName() const { return "optimal_heuristic"; }
};

PlayoutDriver *getPlayoutDriver(std::string name);
