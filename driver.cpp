#include <getopt.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "state.hpp"
#include "player.hpp"

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

void playGameTest(unsigned int numTests) {
  vector<State> ourStates;
  RandomPlayer thePlayer;

  for (unsigned int n = 0; n < numTests; n ++) {
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

    ourStates.push_back(state);
  }

  clock_t t1, t2;
  t1 = clock();
  vector<PlayerId> playoutResults = playouts(ourStates);
  t2 = clock();
  double etime = (double)(t2 - t1) / (double)CLOCKS_PER_SEC;

  int wins[3] = {0, 0, 0};
  for (unsigned int n = 0; n < playoutResults.size(); n ++) {
    switch (playoutResults[n]) {
    case PLAYER_NONE:
      wins[0] ++;
      break;
    case PLAYER_1:
      wins[1] ++;
      break;
    case PLAYER_2:
      wins[2] ++;
      break;
    }
  }

  cout << "GPU Results" << endl;
  cout << "Games drawn: " << wins[0] << endl;
  cout << "Player 1 wins: " << wins[1] << endl;
  cout << "Player 2 wins: " << wins[2] << endl;
  cout << "Elapsed time: " << etime << " seconds" << endl;

  playoutResults.clear();

  t1 = clock();
  for (unsigned int n = 0; n < ourStates.size(); n ++) {
    while (true) {
      if (ourStates[n].isFinished())
        break;

      // Calculate the next move
      Move move = thePlayer.getMove(ourStates[n]);
      // Apply that move to the state
      ourStates[n].move(move);
    }
    playoutResults.push_back(ourStates[n].result());
  }
  t2 = clock();
  etime = (double)(t2 - t1) / (double)CLOCKS_PER_SEC;

  wins[0] = 0; wins[1] = 0; wins[2] = 0;
  for (unsigned int n = 0; n < playoutResults.size(); n ++) {
    switch (playoutResults[n]) {
    case PLAYER_NONE:
      wins[0] ++;
      break;
    case PLAYER_1:
      wins[1] ++;
      break;
    case PLAYER_2:
      wins[2] ++;
      break;
    }
  }

  cout << "CPU Results" << endl;
  cout << "Games drawn: " << wins[0] << endl;
  cout << "Player 1 wins: " << wins[1] << endl;
  cout << "Player 2 wins: " << wins[2] << endl;
  cout << "Elapsed time: " << etime << " seconds" << endl;
}

// printHelp: Output the help message if requested or if there are bad command-line arguments
void printHelp() {
  cout << "Usage: run_ai [--mode|-m single|test] [--playouts|-p N] [--white|-w human|random|mcts] [--black|-n human|random|mcts] [--help]" << endl;
}

int main(int argc, char **argv) {
  srand(2016); // TODO: Should we randomize to time(NULL) instead?
  
  cout << "run_ai : GPU-based checkers player with MCTS" << endl;
  cout << "EE 5351, Fall 2016 Group Project" << endl;
  cout << "Lucas Kramer, Katie Maurer, Ryan Todtleben, and Phil Senum" << endl << endl;

  Player *player1 = NULL;
  Player *player2 = NULL;

  runMode theRunMode = Single;
  unsigned int numTests = 1;

  // Possible options for getopt_long
  static struct option our_options[] = 
    {
      {"mode", required_argument, NULL, 'm'},
      {"playouts", required_argument, NULL, 'p'},
      {"white", required_argument, NULL, 'w'},
      {"black", required_argument, NULL, 'b'},
      {"help", no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

  // Parse the command line options and set up player1 and player2
  int c, option_index;
  bool optionsAllValid = true;
  while ((c = getopt_long(argc, argv, "m:p:w:b:h", our_options, &option_index)) != -1) {
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
    case 'p':
      numTests = atoi(optarg);
      break;
    case 'w':
      if (strcmp(optarg, "human") == 0) {
	player1 = new HumanPlayer;
      }
      else if (strcmp(optarg, "random") == 0) {
	player1 = new RandomPlayer;
      }
      else if (strcmp(optarg, "mcts") == 0) {
	player1 = new MCTSPlayer;
      }
      else {
	cout << "Unrecognized player type '" << optarg << "'" << endl;
	printHelp();
	return 1;
      }
      break;
    case 'b':
      if (strcmp(optarg, "human") == 0) {
	player2 = new HumanPlayer;
      }
      else if (strcmp(optarg, "random") == 0) {
	player2 = new RandomPlayer;
      }
      else if (strcmp(optarg, "mcts") == 0) {
	player2 = new MCTSPlayer;
      }
      else {
	cout << "Unrecognized player type '" << optarg << "'" << endl;
	printHelp();
	return 1;
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
    player1->PrintType();
    cout << " Player 1 ('white') and ";
    player2->PrintType();
    cout << " Player 2 ('black')" << endl;

    Player *players[NUM_PLAYERS] = {player1, player2};
    playGame(players);
  }
  else {
    cout << "Playing " << numTests << " random moves" << endl;
    playGameTest(numTests);
  }

  // Free players as we are done now
  delete player1;
  delete player2;

  return 0;
}
