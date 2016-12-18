#pragma once

#include "state.hpp"

#include <vector>

class PlayoutDriver {
public:
  virtual ~PlayoutDriver() {}
  
  virtual std::vector<PlayerId> runPlayouts(std::vector<State>) const = 0;
  virtual std::string getName() const = 0;
};

// Perform playouts on the CPU from the provided states, returning the winners
class HostPlayoutDriver : public PlayoutDriver {
public:
  ~HostPlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>) const;
  std::string getName() const { return "host"; }
};

// Perform playouts on the GPU with 1 state per block from the provided states,
// returning the winners
class DeviceSinglePlayoutDriver : public PlayoutDriver {
public:
  ~DeviceSinglePlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>) const;
  std::string getName() const { return "device_single"; }
};

// Perform playouts on the GPU with 1 state per tjread from the provided states,
// returning the winners
class DeviceMultiplePlayoutDriver : public PlayoutDriver {
public:
  ~DeviceMultiplePlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>) const;
  std::string getName() const { return "device_multiple"; }
};
