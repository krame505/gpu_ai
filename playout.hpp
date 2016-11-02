#pragma once

#include "state.hpp"

#include <vector>

// Perform number of playouts on the GPU from the provided states, returning the winners
std::vector<Player> playouts(std::vector<State>);


