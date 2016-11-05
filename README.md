# gpu_ai
GPU-based Monte Carlo Tree Search algorithm

## Setup info
This repository is expected to be cloned within the src directory in the NVIDIA GPU SDK.  

Due to the use of C++11 features, gcc 5+ must be used.  
On ECE GPU lab machines, this can be found in /opt/rh/devtoolset-4/root/usr/bin/, which should be added to your PATH so that the correct version is used by make.  

## Rough outline of kernel design
* Block size is 32 (one thread per square on the board)
* Board is represented as a struct containing an array of enums representing the state at each location
* Moves are represented as a struct containing the source and destination locations, as well as arrays of removed and intermediate locations
* The following steps are repeated:
  * Each thread generates the moves for its square
  * The threads copy their moves into a shared 2d array where the first index is which player owns the piece that was moved.  This will involve performing a parallel scan to compute the starting indices
  * Check if any player has no moves, in which case write to the result array that they lost and break out of the loop
  * Each thread picks a random move of the correct turn based on the thread index modulo the number of players
  * Each thread checks if any of the moves picked by lower-index threads interfere with its move
  * The threads whos indices are smaller than that of the first thread to interfere apply their moves to the shared state in parallel
