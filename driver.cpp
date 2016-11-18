#include <getopt.h>
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

// TODO: Some sort of command-line interface to specify player types
int main(int argc, char **argv) {
    static struct option our_options[] = 
    {
	{"black", required_argument, 0, 'r'},
	{"red", required_argument, 0, 'b'},
	{0, 0, 0, 0}
    };

    int c, option_index;
    while ((c = getopt_long(argc, argv, "r:b:", our_options, &option_index)) != -1)
    {
	switch (c)
	{
	case 'b':
	    printf("Black has option %s\n", optarg);
	    break;
	case 'r':
	    printf("Red has option %s\n", optarg);
	    break;
	case '?':
	    break;
	}
    }


  RandomPlayer player1;
  RandomPlayer player2;
  
  Player *players[NUM_PLAYERS] = {&player1, &player2};
  playGame(players);

  return 0;
}
