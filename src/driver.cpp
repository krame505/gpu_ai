#include <vector>
#include <chrono>
#include <functional>
#include <random>
#include <boost/program_options.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "state.hpp"
#include "player.hpp"
#include "genMovesTest.hpp"

#ifdef NDEBUG
#define DEFAULT_NUM_PLAYOUTS 100000

#else
#define DEFAULT_NUM_PLAYOUTS 10

#endif

#define DEFAULT_NUM_GAMES 10

#define NUM_TEST_SETUPS 2
#define SEED 2016
#define MAX_RANDOM_MOVES 100 // NOTE : Is 100 max random moves reasonable?  How long is an average checkers game?

#define NUM_DRAW_MOVES 50

using namespace std;

enum runMode {
  GenMovesTest,
  PlayoutTest,
  GameTest,
  Game
};

istream &operator>>(istream& in, runMode& mode) {
  string token;
  in >> token;
  if (token == "gen_moves_test") {
    mode = GenMovesTest;
  }
  else if (token == "playout_test") {
    mode = PlayoutTest;
  }
  else if (token == "game_test") {
    mode = GameTest;
  }
  else if (token == "game") {
    mode = Game;
  }
  else {
    throw runtime_error("Unknown run mode");
  }

  return in;
}

ostream &operator<<(ostream &os, runMode mode) {
  switch (mode) {
  case GenMovesTest:
    return os << "gen_moves_test";
  case PlayoutTest:
    return os << "playout_test";
  case GameTest:
    return os << "game_test";
  case Game:
    return os << "game";
  default:
    throw runtime_error("Unknown run mode");
  }
}

vector<State> genRandomStates(unsigned int numTests) {
  vector<State> ourStates(numTests);
  RandomPlayer thePlayer;
  default_random_engine generator(SEED);
  uniform_int_distribution<int> distribution(1,MAX_RANDOM_MOVES);
  
  cout << "Building random states..." << endl;
  #pragma omp parallel for
  for (unsigned int n = 0; n < numTests; n++) {
    unsigned int randomMoves = distribution(generator);

    State state = getStartingState();
    
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

void playoutTests(unsigned numTests, PlayoutDriver *playoutDrivers[NUM_TEST_SETUPS]) {
  vector<State> states = genRandomStates(numTests);
  
  // Needed to avoid extra overhead of first kernel call
  cout << "Initializing CUDA context..." << endl;
  cudaDeviceSynchronize();

  for (unsigned i = 0; i < NUM_TEST_SETUPS; i++) {
    cout << "Running test " << i << ": " << playoutDrivers[i]->getName() << "..." << endl;
    playoutTest(states, playoutDrivers[i]);
    cout << endl;
  }
}

// playGame: Implements the game logic.
// Inputs
// players: Array of pointers to Player classes (human, random, or MCTS)
// verbose (optional): Enable verbose output.
// Return: The PlayerId of the winning player (or PLAYER_NONE if there is a draw)
PlayerId playGame(Player *players[NUM_PLAYERS], bool verbose=true) {
  State state = getStartingState();
  unsigned movesSinceLastCapture = 0;

  // Game is over when there are no more possible moves or no captures have been made for an extended time
  while (!state.isGameOver() && movesSinceLastCapture < NUM_DRAW_MOVES) {
    if (verbose) {
      cout << state << endl;
    }
    
    // Get which player's turn it is from the state
    Player *player = players[state.turn];

    // Calculate the next move
    Move move = player->getMove(state, verbose);
    if (verbose) {
      cout << state.turn << " moved " << move << endl;
    }

    // Update the number of moves since the last capture
    if (move.jumps > 0)
      movesSinceLastCapture = 0;
    else
      movesSinceLastCapture++;

    // Apply that move to the state
    state.move(move);
    if (verbose) {
      cout << endl;
    }
  }

  if (verbose) {
    cout << state << endl;
  }

  // Game over: Other player is the winner, unless the game was a draw
  PlayerId result = movesSinceLastCapture < NUM_DRAW_MOVES? state.getNextTurn() : PLAYER_NONE;
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

void gameTests(unsigned numTests, Player *players[NUM_PLAYERS]) {
  int wins[3] = {0, 0, 0};
  for (unsigned i = 0; i < numTests; i++) {
    cout << "Playing game " << i << "... ";
    PlayerId result = playGame(players, false);

    if (result == PLAYER_NONE) {
      cout << "Draw!" << endl;
    }
    else {
      cout << result << " won!" << endl;
    }

    switch (result) {
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
}

int main(int argc, char **argv) {
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
    ("mode,m", boost::program_options::value<runMode>(&theRunMode)->default_value(Game), "run mode\n gen_moves_test playout_test game_test game")
    ("num-playouts,n", boost::program_options::value<unsigned int>(&numTests), "number of playouts")
    ("player1,1", boost::program_options::value<string>(), "player 1 (mode = game, game_test)\nhuman random mcts mcts_host mcts_device_multiple mcts_device_coarse mcts_hybrid mcts_optimal\ntest 1 (mode = playout_test)\nhost device_single device_heuristic device_multiple device_coarse hybrid optimal")
    ("player2,2", boost::program_options::value<string>(), "player 2 (mode = game, game_test)\ntest 2 (mode = playout_test)")
    ("help,h", "print help")
    ;
  boost::program_options::variables_map vm;
  try {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);
    if (vm.count("help")) {
      cout << desc << endl;
      return 0;
    }

    if (!vm.count("num-playouts")) {
      if (theRunMode == GameTest) {
	numTests = DEFAULT_NUM_GAMES;
      }
      else {
	numTests = DEFAULT_NUM_PLAYOUTS;
      }
    }
  
    if (vm.count("player1")) {
      if (theRunMode == Game || theRunMode == GameTest) {
        player1 = getPlayer(vm["player1"].as<string>());
      }
      else {
        playoutDriver1 = getPlayoutDriver(vm["player1"].as<string>());
      }
    }
    else {
      if (theRunMode == Game || theRunMode == GameTest) {
        player1 = new HumanPlayer;
      }
      else {
        playoutDriver1 = new HostPlayoutDriver;
      }
    }    

    if (vm.count("player2")) {
      if (theRunMode == Game || theRunMode == GameTest) {
        player2 = getPlayer(vm["player2"].as<string>());
      }
      else {
        playoutDriver2 = getPlayoutDriver(vm["player2"].as<string>());
      }
    }
    else {
      if (theRunMode == Game || theRunMode == GameTest) {
        player2 = new MCTSPlayer;
      }
      else {
        playoutDriver2 = new DeviceCoarsePlayoutDriver;
      }
    }
  }
  catch (exception& err) {
    cout << err.what() << endl;
    cout << desc << endl;
    return 1;
  }


  // Run the game
  switch (theRunMode) {
  case GenMovesTest:
    cout << "Testing " << numTests << " random states have the same moves on the host and device" << endl;
    cout << endl;

    genMovesTests(numTests);
    break;

  case PlayoutTest:
    cout << "Running " << numTests << " random playouts with ";
    cout << playoutDriver1->getName() << " and ";
    cout << playoutDriver2->getName() << endl;
    cout << endl;

    {
      PlayoutDriver *playoutDrivers[NUM_TEST_SETUPS] = {playoutDriver1, playoutDriver2};
      playoutTests(numTests, playoutDrivers);
    }

    // Free playout drivers as we are done now
    delete playoutDriver1;
    delete playoutDriver2;
    break;

  case GameTest:
    cout << "Playing " << numTests << " games with ";
    cout << player1->getName() << " Player 1 and ";
    cout << player2->getName() << " Player 2" << endl;
    cout << endl;

    {
      Player *players[NUM_PLAYERS] = {player1, player2};
      gameTests(numTests, players);
    }

    // Free players as we are done now
    delete player1;
    delete player2;
    
    break;

  case Game:
    cout << "Playing single game of ";
    cout << player1->getName() << " Player 1 and ";
    cout << player2->getName() << " Player 2" << endl;

    {
      Player *players[NUM_PLAYERS] = {player1, player2};
      playGame(players);
    }

    // Free players as we are done now
    delete player1;
    delete player2;
    break;
  }

  return 0;
}
