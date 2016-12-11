#pragma once

#include "state.hpp"

#include <vector>

// Perform playouts on the CPU from the provided states, returning the winners
std::vector<PlayerId> hostPlayouts(std::vector<State>);

// Perform playouts on the CPU from the provided states applying multiple moves per iteration,
// returning the winners
std::vector<PlayerId> hostPlayoutsFast(std::vector<State>);

// Perform playouts on the GPU from the provided states, returning the winners
std::vector<PlayerId> devicePlayouts(std::vector<State>);

