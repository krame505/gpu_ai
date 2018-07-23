#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <queue>
#include <map>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <memory>

#define MCTS_INITIAL_NUM_PLAYOUTS 50
#define MCTS_NUM_PLAYOUTS_SCALE 0.005
#define MCTS_DEFAULT_TIMEOUT 1 // Seconds
#define MCTS_MAX_RECENT_STATES 15
#define MCTS_DEFAULT_PLAYOUT_DRIVER OptimalHeuristicPlayoutDriver

class Player {
public:
  virtual ~Player() {};

  virtual std::string getName() const = 0;
  virtual Move getMove(const State&, bool verbose=true) = 0;

  virtual void start() {}
  virtual void stop() {}
  virtual void move(const Move&) {}
};

class HumanPlayer : public Player {
public:
  ~HumanPlayer() {};

  std::string getName() const { return "human"; }
  Move getMove(const State&, bool verbose=true);
};

class RandomPlayer : public Player {
public:
  ~RandomPlayer() {};

  std::string getName() const { return "random"; }
  Move getMove(const State&, bool verbose=true);
};

class MCTSPlayer : public Player {
public:
  MCTSPlayer(unsigned initialNumPlayouts,
	     float numPlayoutsScale,
	     unsigned timeout,
	     PlayoutDriver *playoutDriver=new MCTS_DEFAULT_PLAYOUT_DRIVER) :
    initialNumPlayouts(initialNumPlayouts),
    numPlayoutsScale(numPlayoutsScale),
    timeout(timeout),
    playoutDriver(playoutDriver),
    tree(new GameTree(getStartingState())) {}
  MCTSPlayer(PlayoutDriver *playoutDriver=new MCTS_DEFAULT_PLAYOUT_DRIVER) :
    MCTSPlayer(MCTS_INITIAL_NUM_PLAYOUTS,
	       MCTS_NUM_PLAYOUTS_SCALE,
	       MCTS_DEFAULT_TIMEOUT,
	       playoutDriver) {}
  ~MCTSPlayer();

  std::string getName() const {
    return "mcts_" + playoutDriver->getName();
  }
  Move getMove(const State&, bool verbose=true);
  void move(const Move&);
  void start();
  void stop();

private:
  const unsigned initialNumPlayouts;
  const float numPlayoutsScale;
  const unsigned timeout;
  PlayoutDriver *const playoutDriver;
  std::shared_ptr<GameTree> tree;
  std::queue<State> recentStates;
  std::map<State, std::shared_ptr<GameTree>> recentTrees;
  std::mutex treeMutex;
  std::thread workerThread;
  bool updateCancelled = false;
  bool running = false;

  void worker();
};

Player *getPlayer(std::string name);
