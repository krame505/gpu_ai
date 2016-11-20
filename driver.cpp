#include <getopt.h>
#include <cstring>
#include "state.hpp"
#include "player.hpp"

using namespace std;

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

  while (!state.isFinished()) {
    if (verbose) {
      cout << state << endl;
    }
    
    Player *player = players[(unsigned)state.turn];
    Move move = player->getMove(state);
    cout << move << endl;
    state.move(move);
    cout << endl;
  }

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

void printHelp() {
  printf("Usage: run_ai [--red|-r human|random|mcts] [--black|-b human|random|mcts] [--help]\n");
}

int main(int argc, char **argv) {
  printf("run_ai : GPU-based checkers player with MCTS\n");
  printf("EE 5351, Fall 2016 Group Project\n");
  printf("Lucas Kramer, Katie Maurer, Ryan Todtleben, and Phil Senum\n\n");

  Player *player1 = NULL;
  Player *player2 = NULL;
  
  static struct option our_options[] = 
    {
      {"black", required_argument, NULL, 'b'},
      {"red", required_argument, NULL, 'r'},
      {"help", no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

  int c, option_index;
  while ((c = getopt_long(argc, argv, "b:r:h", our_options, &option_index)) != -1)
  {
    switch (c)
    {
    case 'b':
      printf("Player 1 ('black' player) is ");
      if (strcmp(optarg, "human") == 0)
      {
	printf("human\n");
	player1 = new HumanPlayer;
      }
      else if (strcmp(optarg, "random") == 0)
      {
	printf("random\n");
	player1 = new RandomPlayer;
      }
      else if (strcmp(optarg, "mcts") == 0)
      {
	printf("mcts\n");
	player1 = new MCTSPlayer;
      }
      else
      {
	printf("unrecognized type '%s'\n", optarg);
	printHelp();
	return 1;
      }
      break;
    case 'r':
      printf("Player 2 ('red' player) is ");
      if (strcmp(optarg, "human") == 0)
      {
	printf("human\n");
	player2 = new HumanPlayer;
      }
      else if (strcmp(optarg, "random") == 0)
      {
	printf("random\n");
	player2 = new RandomPlayer;
      }
      else if (strcmp(optarg, "mcts") == 0)
      {
	printf("mcts\n");
	player2 = new MCTSPlayer;
      }
      else
      {
	printf("unrecognized type '%s'\n", optarg);
	printHelp();
	return 1;
      }
      break;
    case 'h':
      printHelp();
      return 0;
    case '?':
      break;
    }
  }

  if (player1 == NULL)
  {
    printf("No type specified for player 1 ('black'), defaulting to random\n");
    player1 = new RandomPlayer;
  }
  if (player2 == NULL)
  {
    printf("No type specified for player 2 ('red'), defaulting to random\n");
    player2 = new RandomPlayer;
  }

  Player *players[NUM_PLAYERS] = {player1, player2};
  playGame(players);

  return 0;
}
