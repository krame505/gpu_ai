#pragma once

#include "state.hpp"

#include <vector>

// Perform number of playouts on the CPU from the provided states, returning the winners
std::vector<PlayerId> hostPlayouts(std::vector<State>);

// Perform number of playouts on the GPU from the provided states, returning the winners
std::vector<PlayerId> devicePlayouts(std::vector<State>);

