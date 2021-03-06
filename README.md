# gpu_ai
GPU-based game AI framework utilizing the Monte Carlo Tree Search algorithm

## Setup info
Due to the use of C++11 features, gcc 5+ and CUDA 8.0+ must be used.  This requires setting the  `LD_LIBRARY_PATH` environment variable so that the correct shared libraries can be dynamicly linked.
On ECE GPU lab machines, gcc 5.2 can be found in `/opt/rh/devtoolset-4/root/usr/bin/`, which should be added to your PATH so that the correct version is used by make.  
This can be done automaticly by adding the lines
```
set path=(/opt/rh/devtoolset-4/root/usr/bin $path)

if ($?LD_LIBRARY_PATH) then
 setenv LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
else
 setenv LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64
endif
```
to .cshrc or
```
export PATH=/opt/rh/devtoolset-4/root/usr/bin:$PATH

if [ -z $LD_LIBRARY_PATH ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
fi
```
to .bashrc

## Building and Running

Simply run
```
make
```
or try
```
make nounicode=1
```
if your terminal doesn't print the Unicode checkers correctly.

```
bin/release/run_ai --help
```
Displays all runtime options

To play a game:
```
bin/release/run_ai -1 player1_type -2 player2_type
```
The default is human vs. mcts.  Allowed player types are human, random, mcts, mcts_host, mcts_device_single, mcts_device_heuristic, mcts_device_multiple, mcts_device_coarse, and mcts_hybrid.

Running the human player will enter an interactive mode prompting you to enter a move.  The move format consists of the simply the destination location, or if more than one piece can move to the same location, the source location followed by the destination, seperated by a space.  For ambigous multiple jump moves, the intermediate locations can be entered between the source and destination seperated by spaces.  Promotions are handled automaticly when a checker is moved to the oposite end of the board.  
For example, to move a piece from location b6 to a5, you would type
```b6 a5```
and press enter.  

To run a test:
```
bin/release/run_ai --mode test -1 player1_type -2 player2_type -n num_tests
```
These were used to collect the benchmark results in our report.  Allowed test types are host, device_single, device_heuristic, device_multiple, device_coarse, hybrid, and optimal.
