
#include "player.hpp"

#include <string.h>

#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cassert>
using namespace std;

Move RandomPlayer::getMove(const State &state, bool) {
  vector<Move> moves = state.getMoves();
  return moves[rand() % moves.size()];
}

Move HumanPlayer::getMove(const State &state, bool) {
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

MCTSPlayer::~MCTSPlayer() {
  if (running)
    stop();
  delete playoutDriver;
  delete tree;
};

Move MCTSPlayer::getMove(const State &state, bool verbose) {
  if (state != tree->state) {
    delete tree;
    tree = new GameTree(state);
  }

  sleep(timeout);

  if (verbose) {
    cout << "Tree size: " << tree->getTotalTrials() << endl;
    for (uint8_t i = 0; i < NUM_PLAYERS; i++) {
      PlayerId player = (PlayerId)i;
      cout << player << " score: " << tree->getScore(player) << endl;
    }
  }
  
  return tree->getOptMove(state.turn);
}

void MCTSPlayer::move(const Move &move) {
  treeMutex.lock();
  GameTree *newTree = tree->move(move);
  delete tree;
  tree = newTree;
  updateCancelled = true;
  treeMutex.unlock();
}

void MCTSPlayer::start() {
  assert(!running);
  running = true;
  workerThread = thread(&MCTSPlayer::worker, this);
}

void MCTSPlayer::stop() {
  treeMutex.lock();
  assert(running);
  updateCancelled = true;
  running = false;
  treeMutex.unlock();
  workerThread.join();
}

void MCTSPlayer::worker() {
  treeMutex.lock();
  while (running) {
    unsigned numPlayouts = tree->getTotalTrials() * numPlayoutsScale;
    if (numPlayouts == 0)
      numPlayouts = initialNumPlayouts;
    vector<State> playoutStates = tree->select(numPlayouts);
    treeMutex.unlock();
    vector<PlayerId> results = playoutDriver->runPlayouts(playoutStates);
    treeMutex.lock();
    if (!updateCancelled)
      tree->update(results);
    else
      updateCancelled = false;
  }
  treeMutex.unlock();
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
    return new MCTSPlayer(50, 0, 7, new HostPlayoutDriver);
  }
  else if (name == "mcts_host_heuristic") {
    return new MCTSPlayer(50, 0, 7, new HostHeuristicPlayoutDriver);
  }
  else if (name == "mcts_device_coarse") {
    return new MCTSPlayer(4000, 0.001, 7, new DeviceCoarsePlayoutDriver);
  }
  else if (name == "mcts_device_multiple") {
    return new MCTSPlayer(50, 0.02, 7, new DeviceMultiplePlayoutDriver);
  }
  else if (name == "mcts_hybrid") {
    return new MCTSPlayer(50, 0.02, 7, new HybridPlayoutDriver(1.2));
  }
  else if (name == "mcts_optimal") {
    return new MCTSPlayer(50, 0.004, 7, new OptimalPlayoutDriver);
  }
  else if (name == "mcts_heuristic") {
    return new MCTSPlayer(50, 0.004, 7, new OptimalHeuristicPlayoutDriver);
  }
  else {
    throw runtime_error("Unknown player type");
  }
}
