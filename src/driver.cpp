#include <vector>
#include <chrono>
#include <functional>
#include <memory>
#include <random>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "cxxopts.hpp"

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

vector<PlayerId> playoutTest(const vector<State> &states, unique_ptr<PlayoutDriver> playoutDriver) {
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

void playoutTests(unsigned numTests, unique_ptr<PlayoutDriver> playoutDrivers[NUM_TEST_SETUPS]) {
  vector<State> states = genRandomStates(numTests);
  
  // Needed to avoid extra overhead of first kernel call
  cout << "Initializing CUDA context..." << endl;
  cudaDeviceSynchronize();

  vector<PlayerId> results[NUM_TEST_SETUPS];
  for (unsigned i = 0; i < NUM_TEST_SETUPS; i++) {
    cout << "Running test " << i << ": " << playoutDrivers[i]->getName() << "..." << endl;
    results[i] = playoutTest(states, move(playoutDrivers[i]));
    cout << endl;
  }

  unsigned numSame = 0;
  for (unsigned i = 0; i < numTests; i++) {
    if (results[0][i] == results[1][i])
      numSame++;
  }

  cout << "Similar results: " << (float)numSame / numTests * 100 << "%" << endl;
}

// playGame: Implements the game logic.
// Inputs
// players: Array of pointers to Player classes (human, random, or MCTS)
// verbose (optional): Enable verbose output.
// Return: The PlayerId of the winning player (or PLAYER_NONE if there is a draw)
PlayerId playGame(unique_ptr<Player> players[NUM_PLAYERS], bool verbose=true) {
  State state = getStartingState();

  // Start all players
  for (unsigned i = 0; i < NUM_PLAYERS; i++) {
    players[i]->start();
  }

  // Game is over when there are no more possible moves or no captures have been made for an extended time
  while (!state.isGameOver()) {
    if (verbose) {
      cout << state << endl;
    }
    
    // Get which player's turn it is from the state
    const unique_ptr<Player> &player = players[state.turn];

    // Calculate the next move
    Move move = player->getMove(state, verbose);
    if (verbose) {
      cout << state.turn << " moved " << move << endl;
    }

    // Notify all players of the move
    for (unsigned i = 0; i < NUM_PLAYERS; i++) {
      players[i]->move(move);
    }

    // Apply that move to the state
    state.move(move);
    if (verbose) {
      cout << endl;
    }
  }

  if (verbose) {
    cout << state << endl;
  }

  // Stop all players
  for (unsigned i = 0; i < NUM_PLAYERS; i++) {
    players[i]->stop();
  }

  // Game over: Other player is the winner, unless the game was a draw
  PlayerId result = state.getWinner();
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

void gameTests(unsigned numTests, unique_ptr<Player> players[NUM_PLAYERS]) {
  int wins[3] = {0, 0, 0};
  for (unsigned i = 0; i < numTests; i++) {
    cout << "Playing game " << i << "... " << flush;
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
  runMode mode = Game;
  unique_ptr<Player> player1 = NULL;
  unique_ptr<Player> player2 = NULL;
  unique_ptr<PlayoutDriver> playoutDriver1;
  unique_ptr<PlayoutDriver> playoutDriver2;

  cxxopts::Options options(argv[0]);
  options.add_options()
    ("m,mode", "run mode: gen_moves_test playout_test game_test game",
     cxxopts::value<runMode>(mode))
    ("n,num_tests", "number of tests to run",
     cxxopts::value<unsigned int>(numTests))
    ("1,player1", "positional arg 1                                       * player 1 (mode = game, game_test): human random mcts mcts_host mcts_host_heuristic mcts_device_multiple mcts_device_coarse mcts_hybrid mcts_optimal mcts_heuristic * playout driver 1 (mode = playout_test): host host_heuristic device_single device_heuristic device_multiple device_coarse hybrid hybrid_heruistic optimal optimal_heuristic",
     cxxopts::value<string>())
    ("2,player2", "positional arg 2                                       * player 2 (mode = game, game_test)                   * test 2   (mode = playout_test)",
     cxxopts::value<string>())
    ("h,help", "print help")
    ;

  options.parse_positional(vector<string>{"player1", "player2"});

    
  try {
    options.parse(argc, argv);

    if (options.count("help")) {
      cout << options.help() << endl;
      return 0;
    }

    if (!options.count("num_tests")) {
      if (mode == GameTest) {
	numTests = DEFAULT_NUM_GAMES;
      }
      else {
	numTests = DEFAULT_NUM_PLAYOUTS;
      }
    }
  
    if (options.count("player1")) {
      if (mode == Game || mode == GameTest) {
        player1 = getPlayer(options["player1"].as<string>());
      }
      else {
        playoutDriver1 = getPlayoutDriver(options["player1"].as<string>());
      }
    }
    else {
      if (mode == Game || mode == GameTest) {
        player1 = make_unique<HumanPlayer>();
      }
      else {
        playoutDriver1 = make_unique<HostPlayoutDriver>();
      }
    }    

    if (options.count("player2")) {
      if (mode == Game || mode == GameTest) {
        player2 = getPlayer(options["player2"].as<string>());
      }
      else {
        playoutDriver2 = getPlayoutDriver(options["player2"].as<string>());
      }
    }
    else {
      if (mode == Game || mode == GameTest) {
        player2 = make_unique<MCTSPlayer>();
      }
      else {
        playoutDriver2 = make_unique<DeviceCoarsePlayoutDriver>();
      }
    }
  }
  catch (exception& err) {
    cout << err.what() << endl;
    cout << options.help() << endl;
    return 1;
  }


  // Run the game
  switch (mode) {
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
      unique_ptr<PlayoutDriver> playoutDrivers[NUM_TEST_SETUPS] = {move(playoutDriver1), move(playoutDriver2)};
      playoutTests(numTests, playoutDrivers);
    }
    break;

  case GameTest:
    cout << "Playing " << numTests << " games with ";
    cout << player1->getName() << " Player 1 and ";
    cout << player2->getName() << " Player 2" << endl;
    cout << endl;
    {
      unique_ptr<Player> players[NUM_PLAYERS] = {move(player1), move(player2)};
      gameTests(numTests, players);
    }
    break;

  case Game:
    cout << "Playing single game of ";
    cout << player1->getName() << " Player 1 and ";
    cout << player2->getName() << " Player 2" << endl;
    {
      unique_ptr<Player> players[NUM_PLAYERS] = {move(player1), move(player2)};
      playGame(players);
    }
    break;
  }

  return 0;
}
