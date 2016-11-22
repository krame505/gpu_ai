#include <getopt.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
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
    Player *player = players[(unsigned)state.turn];
    // Calculate the next move
    Move move = player->getMove(state);
    cout << move << endl;
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

void playGameTest(unsigned int numTests, unsigned int randomMoves)
{
  clock_t t1, t2;
  t1 = clock();

  int wins[3] = {0, 0, 0};

  for (unsigned int n = 0; n < numTests; n ++)
  {
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

    Player *player1 = new RandomPlayer;
    Player *player2 = new RandomPlayer;
    Player *players[NUM_PLAYERS] = {player1, player2};

    for (unsigned int m = 0; m < randomMoves; m ++)
    {
      if (state.isFinished())
	break;

      // Get which player's turn it is from the state
      Player *player = players[(unsigned)state.turn];
      // Calculate the next move
      Move move = player->getMove(state);
      // Apply that move to the state
      state.move(move);
    }

    delete player1;
    delete player2;
   
    // Then switch to MCTS for the rest of the game (what we really want to test)

    player1 = new MCTSPlayer;
    player2 = new MCTSPlayer;
    players[0] = player1;
    players[1] = player2;

    while (!state.isFinished())
    {
      // Get which player's turn it is from the state
      Player *player = players[(unsigned)state.turn];
      // Calculate the next move
      Move move = player->getMove(state);
      // Apply that move to the state
      state.move(move);
    }

    // Tally the results
    
    PlayerId result = state.result();
    switch (result)
    {
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

    delete player1;
    delete player2;
  }

  // Calculate the total running time
  t2 = clock();
  double etime = (double)(t2 - t1) / (double)CLOCKS_PER_SEC;

  cout << "Games drawn: " << wins[0] << endl;
  cout << "Player 1 wins: " << wins[1] << endl;
  cout << "Player 2 wins: " << wins[2] << endl;
  cout << "Elapsed time: " << etime << " seconds" << endl;
}

// printHelp: Output the help message if requested or if there are bad command-line arguments
void printHelp() {
  cout << "Usage: run_ai [--mode|-m single|test] [--playouts|-p N] [--random|-R N] [--red|-r human|random|mcts] [--black|-b human|random|mcts] [--help]" << endl;
}

int main(int argc, char **argv) {
  cout << "run_ai : GPU-based checkers player with MCTS" << endl;
  cout << "EE 5351, Fall 2016 Group Project" << endl;
  cout << "Lucas Kramer, Katie Maurer, Ryan Todtleben, and Phil Senum" << endl << endl;

  Player *player1 = NULL;
  Player *player2 = NULL;

  runMode theRunMode = Single;
  unsigned int numTests = 1, randomMoves = 0;

  // Possible options for getopt_long
  static struct option our_options[] = 
    {
      {"mode", required_argument, NULL, 'm'},
      {"playouts", required_argument, NULL, 'p'},
      {"random", required_argument, NULL, 'R'},
      {"black", required_argument, NULL, 'b'},
      {"red", required_argument, NULL, 'r'},
      {"help", no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

  // Parse the command line options and set up player1 and player2
  int c, option_index;
  bool optionsAllValid = true;
  while ((c = getopt_long(argc, argv, "m:p:R:b:r:h", our_options, &option_index)) != -1)
  {
    switch (c)
    {
    case 'm':
      if (strcmp(optarg, "single") == 0)
      {
	theRunMode = Single;
      }
      else if (strcmp(optarg, "test") == 0)
      {
	theRunMode = Test;
      }
      else
      {
	cout << "Unrecognized run mode '" << optarg << "'" << endl;
	printHelp();
	return 1;
      }
      break;
    case 'p':
      numTests = atoi(optarg);
      break;
    case 'R':
      randomMoves = atoi(optarg);
      break;
    case 'b':
      if (strcmp(optarg, "human") == 0)
      {
	player1 = new HumanPlayer;
      }
      else if (strcmp(optarg, "random") == 0)
      {
	player1 = new RandomPlayer;
      }
      else if (strcmp(optarg, "mcts") == 0)
      {
	player1 = new MCTSPlayer;
      }
      else
      {
	cout << "Unrecognized player type '" << optarg << "'" << endl;
	printHelp();
	return 1;
      }
      break;
    case 'r':
      if (strcmp(optarg, "human") == 0)
      {
	player2 = new HumanPlayer;
      }
      else if (strcmp(optarg, "random") == 0)
      {
	player2 = new RandomPlayer;
      }
      else if (strcmp(optarg, "mcts") == 0)
      {
	player2 = new MCTSPlayer;
      }
      else
      {
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
  if (!optionsAllValid)
  {
    printHelp();
    return 1;
  }

  // Run the game
  if (theRunMode == Single)
  {
    // Assume random players if not otherwise specified
    if (player1 == NULL)
    {
      player1 = new RandomPlayer;
    }
    if (player2 == NULL)
    {
      player2 = new RandomPlayer;
    }

    cout << "Playing single game of ";
    player1->PrintType();
    cout << " Player 1 ('black') and ";
    player2->PrintType();
    cout << " Player 2 ('red')" << endl;

    Player *players[NUM_PLAYERS] = {player1, player2};
    playGame(players);
  }
  else
  {
    cout << "Playing " << numTests << " games with " << randomMoves << " random opening moves each" << endl;
    playGameTest(numTests, randomMoves);
  }

  // Free players as we are done now
  delete player1;
  delete player2;

  return 0;
}
