#include <getopt.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <vector>
#include <chrono>
#include <functional>
#include <boost/program_options.hpp>

#include "state.hpp"
#include "player.hpp"
#include "genMovesTest.hpp"

#ifdef NDEBUG
#define DEFAULT_NUM_PLAYOUTS 100000

#else
#define DEFAULT_NUM_PLAYOUTS 10

#endif

#define NUM_TEST_SETUPS 2

using namespace std;

enum runMode {
  GenMovesTest,
  Test,
  Single
};

istream &operator>>(istream& in, runMode& mode) {
  string token;
  in >> token;
  if (token == "gen_moves_test") {
    mode = GenMovesTest;
  }
  else if (token == "test") {
    mode = Test;
  }
  else if (token == "single") {
    mode = Single;
  }
  else {
    //throw boost::program_options::validation_error(boost::program_options::validation_error::invalid_option_value);
    throw runtime_error("Unknown run mode");
  }

  return in;
}

ostream &operator<<(ostream &os, runMode mode) {
  switch (mode) {
  case GenMovesTest:
    return os << "gen_moves_test";
  case Test:
    return os << "test";
  case Single:
    return os << "single";
  default:
    assert(false);
    return os; // Unreachable, but to make -Wall shut up
  }
}


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
  while (!state.isGameOver()) {
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

  // Game over: Other player is the winner
  PlayerId result = state.getNextTurn();
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

vector<State> genRandomStates(unsigned int numTests) {
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
      if (state.isGameOver())
	break;

      // Calculate the next move
      Move move = thePlayer.getMove(state);
      // Apply that move to the state
      state.move(move);
    }

    ourStates[n] = state;
  }

  return ourStates;
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
  vector<State> states = genRandomStates(numTests);
  
  for (unsigned int i = 0; i < NUM_TEST_SETUPS; i++) {
    cout << "Running test " << i << ": " << playoutDrivers[i]->getName() << "..." << endl;
    playoutTest(states, playoutDrivers[i]);
    cout << endl;
  }
}

void genMovesTests(unsigned int numTests) {
  vector<State> states = genRandomStates(numTests);
  
  cout << "Testing genMoves host and device results are the same..." << endl;
  for (State state : states) {
    if (!genMovesTest(state)) {
      cout << "Failed" << endl;
      exit(1);
    }
  }
  cout << "Passed" << endl;
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
    //throw boost::program_options::validation_error(boost::program_options::validation_error::invalid_option_value);
    throw runtime_error("Unknown player type");
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
    //throw boost::program_options::validation_error(boost::program_options::validation_error::invalid_option_value);
    throw runtime_error("Unknown playout type");
  }
}

int main(int argc, char **argv) {
  srand(2016); // TODO: Should we randomize to time(NULL) instead?

  cout << "run_ai : GPU-based checkers player with MCTS" << endl;
  cout << "EE 5351, Fall 2016 Group Project" << endl;
  cout << "Lucas Kramer, Katie Maurer, Ryan Todtleben, and Phil Senum" << endl;

  unsigned int numTests;
  runMode theRunMode;
  Player *player1 = NULL;
  Player *player2 = NULL;
  PlayoutDriver *playoutDriver1 = NULL;
  PlayoutDriver *playoutDriver2 = NULL;

  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("mode,m", boost::program_options::value<runMode>(&theRunMode)->default_value(Single), "run mode")
    ("num-playouts,n", boost::program_options::value<unsigned int>(&numTests)->default_value(DEFAULT_NUM_PLAYOUTS), "number of playouts")
    ("player1,1", boost::program_options::value<string>(), "player 1")
    ("player2,2", boost::program_options::value<string>(), "player 2")
    ("help,h", "print help")
    ;
  boost::program_options::variables_map vm;
  try {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);
    if (vm.count("help")) {
      cout << desc << endl;
      return 1;
    }
  
    if (vm.count("player1")) {
      if (theRunMode == Single) {
	player1 = getPlayer(vm["player1"].as<string>());
      }
      else {
	playoutDriver1 = getPlayoutDriver(vm["player1"].as<string>());
      }
    }
    else {
      if (theRunMode == Single) {
	player1 = new RandomPlayer;
      }
      else {
	playoutDriver1 = new HostPlayoutDriver;
      }
    }    

    if (vm.count("player2")) {
      if (theRunMode == Single) {
	player2 = getPlayer(vm["player2"].as<string>());
      }
      else {
	playoutDriver2 = getPlayoutDriver(vm["player2"].as<string>());
      }
    }
    else {
      if (theRunMode == Single) {
	player2 = new RandomPlayer;
      }
      else {
	playoutDriver2 = new DevicePlayoutDriver;
      }
    }
  }
  catch (exception& err) {
    cout << err.what() << endl;
    cout << desc << endl;
    return 1;
  }


  // Run the game
  if (theRunMode == Single) {
    cout << "Playing single game of ";
    cout << player1->getName() << " Player 1 and ";
    cout << player2->getName() << " Player 2" << endl;

    Player *players[NUM_PLAYERS] = {player1, player2};
    playGame(players);

    // Free players as we are done now
    delete player1;
    delete player2;
  }
  else if (theRunMode == Test) {
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
  else if (theRunMode == GenMovesTest) {
    cout << "Testing " << numTests << " random states have the same moves on the host and device" << endl;
    cout << endl;

    genMovesTests(numTests);
  }

  return 0;
}
