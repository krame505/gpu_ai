#include <getopt.h>
#include <cstring>
#include "state.hpp"
#include "player.hpp"

using namespace std;

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

// printHelp: Output the help message if requested or if there are bad command-line arguments
void printHelp() {
  cout << "Usage: run_ai [--red|-r human|random|mcts] [--black|-b human|random|mcts] [--help]" << endl;
}

int main(int argc, char **argv) {
  cout << "run_ai : GPU-based checkers player with MCTS" << endl;
  cout << "EE 5351, Fall 2016 Group Project" << endl;
  cout << "Lucas Kramer, Katie Maurer, Ryan Todtleben, and Phil Senum" << endl << endl;

  Player *player1 = NULL;
  Player *player2 = NULL;

  // Possible options for getopt_long
  static struct option our_options[] = 
    {
      {"black", required_argument, NULL, 'b'},
      {"red", required_argument, NULL, 'r'},
      {"help", no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

  // Parse the command line options and set up player1 and player2
  int c, option_index;
  bool optionsAllValid = true;
  while ((c = getopt_long(argc, argv, "b:r:h", our_options, &option_index)) != -1)
  {
    switch (c)
    {
    case 'b':
      cout << "Player 1 ('black' player) is ";
      if (strcmp(optarg, "human") == 0)
      {
	cout << "human" << endl;
	player1 = new HumanPlayer;
      }
      else if (strcmp(optarg, "random") == 0)
      {
	cout << "random" << endl;
	player1 = new RandomPlayer;
      }
      else if (strcmp(optarg, "mcts") == 0)
      {
	cout << "mcts" << endl;
	player1 = new MCTSPlayer;
      }
      else
      {
	cout << "unrecognized type '" << optarg << "'" << endl;
	printHelp();
	return 1;
      }
      break;
    case 'r':
      cout << "Player 2 ('red' player) is ";
      if (strcmp(optarg, "human") == 0)
      {
	cout << "human" << endl;
	player2 = new HumanPlayer;
      }
      else if (strcmp(optarg, "random") == 0)
      {
	cout << "random" << endl;
	player2 = new RandomPlayer;
      }
      else if (strcmp(optarg, "mcts") == 0)
      {
	cout << "mcts" << endl;
	player2 = new MCTSPlayer;
      }
      else
      {
	cout << "unrecognized type '" << optarg << "'" << endl;
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

  // Assume random players if not otherwise specified
  if (player1 == NULL)
  {
    cout << "No type specified for player 1 ('black'), defaulting to random" << endl;
    player1 = new RandomPlayer;
  }
  if (player2 == NULL)
  {
    cout << "No type specified for player 2 ('red'), defaulting to random" << endl;
    player2 = new RandomPlayer;
  }

  // Run the game
  Player *players[NUM_PLAYERS] = {player1, player2};
  playGame(players);

  return 0;
}
