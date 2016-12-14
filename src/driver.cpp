#include <getopt.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <vector>
#include <chrono>
#include <functional>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "state.hpp"
#include "player.hpp"
#include "genMovesTest.hpp"

#ifdef NDEBUG
#define DEFAULT_NUM_PLAYOUTS 1000

#else
#define DEFAULT_NUM_PLAYOUTS 10

#endif

#define NUM_TEST_SETUPS 2

using namespace std;

enum runMode {
  Test,
  Single
};

// playGame: Implements the game logic.
// Inputs
// players: Array of pointers to Player classes (human, random, or MCTS)
// verbose (optional): Enable verbose output.
// Return: The PlayerId of the winning player (or PLAYER_NONE if there is a draw)
PlayerId playGame(Player *players[NUM_PLAYERS], bool verbose=true) {
  // Unspecified BoardItems are initialized to 0
  State state = {
    {{{}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}},
     {{true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}},
     {{}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}},
     {{}, {}, {}, {}, {}, {}, {}, {}},
     {{}, {}, {}, {}, {}, {}, {}, {}},
     {{true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}},
     {{}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}},
     {{true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}}},
    PLAYER_1
  };

  // Game is over when there are no more possible moves
  while (!state.isFinished()) {
    if (verbose) {
      cout << state << endl;
    }
    
    // Get which player's turn it is from the state
    Player *player = players[state.turn];
    // Calculate the next move
    Move move = player->getMove(state);
    cout << state.turn << " moved " << move << endl;
    // Apply that move to the state
    state.move(move);
    cout << endl;
  }

  if (verbose) {
    cout << state << endl;
  }

  // Game over: check the state to see who won the game
  PlayerId result = state.result();
  if (verbose) {
    if (result == PLAYER_NONE) {
      cout << "Draw!" << endl;
    }
    else {
      cout << result << " won!" << endl;
    }
  }
  return result;
}

vector<PlayerId> playoutTest(const vector<State> &states, PlayoutDriver *playoutDriver) {
  auto t1 = chrono::high_resolution_clock::now();
  vector<PlayerId> playoutResults = playoutDriver->runPlayouts(states);
  auto t2 = chrono::high_resolution_clock::now();
  chrono::duration<double> diff = t2 - t1;

  int wins[3] = {0, 0, 0};
  for (unsigned int n = 0; n < playoutResults.size(); n++) {
    switch (playoutResults[n]) {
    case PLAYER_NONE:
      wins[0]++;
      break;
    case PLAYER_1:
      wins[1]++;
      break;
    case PLAYER_2:
      wins[2]++;
      break;
    }
  }

  cout << "=== Results ===" << endl;
  cout << "Games drawn: " << wins[0] << endl;
  cout << "Player 1 wins: " << wins[1] << endl;
  cout << "Player 2 wins: " << wins[2] << endl;
  cout << "Elapsed time: " << diff.count() << " seconds" << endl;

  return playoutResults;
}

void playoutTests(unsigned int numTests, PlayoutDriver *playoutDrivers[NUM_TEST_SETUPS]) {
  vector<State> ourStates(numTests);
  RandomPlayer thePlayer;

  cout << "Building random states..." << endl;
  #pragma omp parallel for
  for (unsigned int n = 0; n < numTests; n++) {
    unsigned int randomMoves = rand() % 100; // TODO : Is 100 max random moves reasonable?  How long is an average checkers game?

    State state = {
      {{{}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}},
       {{true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}},
       {{}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}},
       {{}, {}, {}, {}, {}, {}, {}, {}},
       {{}, {}, {}, {}, {}, {}, {}, {}},
       {{true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}},
       {{}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}},
       {{true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}}},
      PLAYER_1
    };

    // Carry out the opening random moves
    for (unsigned int m = 0; m < randomMoves; m ++) {
      if (state.isFinished())
	break;

      // Calculate the next move
      Move move = thePlayer.getMove(state);
      // Apply that move to the state
      state.move(move);
    }

    ourStates[n] = state;
  }

#ifndef NDEBUG
  cout << "Testing genMoves host and device results are the same..." << endl;
  for (State state : ourStates) {
    assert(genMovesTest(state));
  }
#endif

  for (unsigned int i = 0; i < NUM_TEST_SETUPS; i++) {
    cout << "Running test " << i << ": " << playoutDrivers[i]->getName() << "..." << endl;
    playoutTest(ourStates, playoutDrivers[i]);
    cout << endl;
  }
}

// printHelp: Output the help message if requested or if there are bad command-line arguments
void printHelp() {
  cout << "Usage: run_ai [--white|-w|-1 human|random|mcts] [--black|-b|-2 human|random|mcts] [--mode|-m single|test] [--num-playouts|-n N] [--help|-h]" << endl;
}

Player *getPlayer(string name) {
  if (name == "human") {
    return new HumanPlayer;
  }
  else if (name == "random") {
    return new RandomPlayer;
  }
  else if (name == "mcts") {
    return new MCTSPlayer;
  }
  else if (name == "mcts_host") {
    return new MCTSPlayer(5000, 7, new HostPlayoutDriver);
  }
  else if (name == "mcts_host1") {
    return new MCTSPlayer(50, 7, new HostPlayoutDriver);
  }
  else if (name == "mcts_device") {
    return new MCTSPlayer(new DevicePlayoutDriver);
  }
  else {
    cout << "Unrecognized player type '" << name << "'" << endl;
    printHelp();
    exit(1);
  }
}

PlayoutDriver *getPlayoutDriver(string name) {
  if (name == "device") {
    return new DevicePlayoutDriver;
  }
  else if (name == "host") {
    return new HostPlayoutDriver;
  }
  else if (name == "host_fast") {
    return new HostFastPlayoutDriver;
  }
  else {
    cout << "Unrecognized playout type '" << name << "'" << endl;
    printHelp();
    exit(1);
  }
}

int main(int argc, char **argv) {
  srand(2016); // TODO: Should we randomize to time(NULL) instead?

  int driverVersion;
  cuDriverGetVersion(&driverVersion);
  
  cout << "run_ai : GPU-based checkers player with MCTS" << endl;
  cout << "EE 5351, Fall 2016 Group Project" << endl;
  cout << "Lucas Kramer, Katie Maurer, Ryan Todtleben, and Phil Senum" << endl;
  cout << "CUDA Version " << CUDA_VERSION << " and Runtime Version " << CUDART_VERSION << endl;
  cout << "Driver Version " << driverVersion << endl << endl;
  
  Player *player1 = NULL;
  Player *player2 = NULL;

  PlayoutDriver *playoutDriver1 = NULL;
  PlayoutDriver *playoutDriver2 = NULL;

  runMode theRunMode = Single;
  unsigned int numTests = DEFAULT_NUM_PLAYOUTS;

  // Possible options for getopt_long
  static struct option our_options[] = 
    {
      {"mode", required_argument, NULL, 'm'},
      {"num-playouts", required_argument, NULL, 'n'},
      {"white", required_argument, NULL, 'w'},
      {"black", required_argument, NULL, 'b'},
      {"help", no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

  // Parse the command line options and set up player1 and player2
  int c, option_index;
  bool optionsAllValid = true;
  while ((c = getopt_long(argc, argv, "m:n:w:b:1:2:h", our_options, &option_index)) != -1) {
    switch (c) {
    case 'm':
      if (strcmp(optarg, "single") == 0) {
	theRunMode = Single;
      }
      else if (strcmp(optarg, "test") == 0) {
	theRunMode = Test;
      }
      else {
	cout << "Unrecognized run mode '" << optarg << "'" << endl;
	printHelp();
	return 1;
      }
      break;
    case 'n':
      numTests = atoi(optarg);
      break;
    case 'w':
    case '1':
      switch (theRunMode) {
      case Single:
	player1 = getPlayer(string(optarg));
	break;
      case Test:
	playoutDriver1 = getPlayoutDriver(string(optarg));
	break;
      }
      break;
    case 'b':
    case '2':
      switch (theRunMode) {
      case Single:
	player2 = getPlayer(string(optarg));
	break;
      case Test:
	playoutDriver2 = getPlayoutDriver(string(optarg));
	break;
      }
      break;
    case 'h':
      printHelp();
      return 0;
    case '?':
      optionsAllValid = false;
      break;
    }
  }

  // If an invalid option was passed, print help and exit
  if (!optionsAllValid) {
    printHelp();
    return 1;
  }

  // Run the game
  if (theRunMode == Single) {
    // Assume random players if not otherwise specified
    if (player1 == NULL) {
      player1 = new RandomPlayer;
    }
    if (player2 == NULL) {
      player2 = new RandomPlayer;
    }

    cout << "Playing single game of ";
    cout << player1->getName() << " Player 1 ('white') and ";
    cout << player2->getName() << " Player 2 ('black')" << endl;

    Player *players[NUM_PLAYERS] = {player1, player2};
    playGame(players);

    // Free players as we are done now
    delete player1;
    delete player2;
  }
  else {
    // Assume host and device playouts if not otherwise specified
    if (player1 == NULL) {
      playoutDriver1 = new HostPlayoutDriver;
    }
    if (player2 == NULL) {
      playoutDriver2 = new DevicePlayoutDriver;
    }

    cout << "Running " << numTests << " random playouts with ";
    cout << playoutDriver1->getName() << " and ";
    cout << playoutDriver2->getName() << endl;
    cout << endl;

    PlayoutDriver *playoutDrivers[NUM_TEST_SETUPS] = {playoutDriver1, playoutDriver2};
    playoutTests(numTests, playoutDrivers);

    // Free playout drivers as we are done now
    delete playoutDriver1;
    delete playoutDriver2;
  }

  return 0;
}
