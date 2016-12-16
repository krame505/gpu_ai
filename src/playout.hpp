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

// Perform playouts on the CPU from the provided states applying multiple moves per iteration,
// returning the winners
class HostFastPlayoutDriver : public PlayoutDriver {
public:
  ~HostFastPlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>) const;
  std::string getName() const { return "host_fast"; }
};

// Perform playouts on the GPU from the provided states, returning the winners
class DevicePlayoutDriver : public PlayoutDriver {
public:
  ~DevicePlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>) const;
  std::string getName() const { return "device"; }
};

// Perform playouts on the GPU from the provided states, returning the winners
class DeviceSimplePlayoutDriver : public PlayoutDriver {
public:
  ~DeviceSimplePlayoutDriver() {};

  std::vector<PlayerId> runPlayouts(std::vector<State>) const;
  std::string getName() const { return "device_simple"; }
};
