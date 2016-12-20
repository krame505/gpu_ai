
#include "player.hpp"

#include <string.h>

#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
using namespace std;

Move RandomPlayer::getMove(const State &state) {
  vector<Move> moves = state.getMoves();
  return moves[rand() % moves.size()];
}

Move HumanPlayer::getMove(const State &state) {
  char input[100];

  while (true) {
    string error;
    vector<Move> moves = state.getMoves();

    cout << "Valid moves:" << endl;
    for (Move &m : moves) {
      cout << m << endl;
    }

    cout << "Move for " << state.turn << ": ";
    cin.getline(input, 100);

    unsigned len = strlen(input);

    unsigned i = 0;
    Loc to(BOARD_SIZE, BOARD_SIZE); // Out of bounds

    if (len == 1 || len == 3 || len > 5) {
      error = "Invalid move syntax";
      moves.clear();
    }

    if (len > 2) {
      Loc from(BOARD_SIZE - (input[1] - '0'), input[0] - 'a');
      if (from.row >= BOARD_SIZE || from.col >= BOARD_SIZE) {
        error = "Invalid source location";
        moves.clear();
      }
      for (unsigned int n = 0; n < moves.size(); n++) {
        if (moves[n].from.row != from.row || moves[n].from.col != from.col) {
          moves.erase(moves.begin() + n);
          n--;
        }
      }
      i += 2;
      if (input[i] == ' ')
        i++;
    }

    if (i < len) {
      to = Loc(BOARD_SIZE - (input[i + 1] - '0'), input[i] - 'a');
      if (to.row >= BOARD_SIZE || to.col >= BOARD_SIZE) {
        error = "Invalid destination";
        moves.clear();
      }
      for (unsigned int n = 0; n < moves.size(); n++) {
        if (moves[n].to.row != to.row || moves[n].to.col != to.col) {
          moves.erase(moves.begin() + n);
          n--;
        }
      }
    }

    if (moves.size() == 0) {
      if (error.size())
        cout << error << ", try again" << endl;
      else
        cout << "Invalid move, try again" << endl;
    }
    else if (moves.size() > 1) {
      cout << "Ambiguous move" << endl;
    }
    else {
      return moves[0];
    }
  }
}

Move MCTSPlayer::getMove(const State &state) {
  // Don't bother running MCTS if there is only 1 possible move
  vector<Move> moves = state.getMoves();
  if (moves.size() == 1)
    return moves[0];

  GameTree *tree = new GameTree(state);

  // Build the tree
  auto start = chrono::high_resolution_clock::now();
  double elapsedTime;
  unsigned iterations = 0;
  do {
    tree->expand(numPlayouts, playoutDriver);

    auto current = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = current - start;
    elapsedTime = diff.count();

    iterations++;
  } while (elapsedTime < timeout);

  if (iterations > targetIterations)
    numPlayouts *= MCTS_NUM_PLAYOUTS_SCALE;
  else if (iterations < targetIterations && numPlayouts > 2)
    numPlayouts /= MCTS_NUM_PLAYOUTS_SCALE;

#ifdef VERBOSE
  cout << "Finished " << iterations << " iterations" << endl;
  cout << "Next iteration with " << numPlayouts << " playouts" << endl;
  cout << "Time: " << elapsedTime << " seconds" << endl;
  for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
    PlayerId player = (PlayerId)i;
    cout << player << " score: " << tree->getScore(player) << endl;
  }
#endif

  Move move = tree->getOptMove(state.turn);

  delete tree;
  return move;
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
    return new MCTSPlayer(100, 1000, 7, new HostPlayoutDriver);
  }
  else if (name == "mcts_device_single") {
    return new MCTSPlayer(new DeviceSinglePlayoutDriver);
  }
  else if (name == "mcts_device_heuristic") {
    return new MCTSPlayer(new DeviceHeuristicPlayoutDriver);
  }
  else if (name == "mcts_device_multiple") {
    return new MCTSPlayer(5000, 20, 7, new DeviceMultiplePlayoutDriver);
  }
  else if (name == "mcts_device_relaunch") {
    return new MCTSPlayer(new DeviceRelaunchPlayoutDriver);
  }
  else if (name == "mcts_hybrid") {
    return new MCTSPlayer(10000, 40, 7, new HybridPlayoutDriver(6));
  }
  else {
    throw runtime_error("Unknown player type");
  }
}
