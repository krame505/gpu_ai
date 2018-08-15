#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <memory>

#define MCTS_INITIAL_NUM_PLAYOUTS 50
#define MCTS_NUM_PLAYOUTS_SCALE 0.003
#define MCTS_DEFAULT_TIMEOUT 10 // Seconds
#define MCTS_MAX_RECENT_STATES 15
#define MCTS_DEFAULT_PLAYOUT_DRIVER OptimalPlayoutDriver

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
	     std::unique_ptr<PlayoutDriver> playoutDriver=std::make_unique<MCTS_DEFAULT_PLAYOUT_DRIVER>()) :
    initialNumPlayouts(initialNumPlayouts),
    numPlayoutsScale(numPlayoutsScale),
    timeout(timeout),
    playoutDriver(std::move(playoutDriver)),
    tree(std::make_shared<GameTree>(getStartingState())) {}
  MCTSPlayer(std::unique_ptr<PlayoutDriver> playoutDriver=std::make_unique<MCTS_DEFAULT_PLAYOUT_DRIVER>()) :
    MCTSPlayer(MCTS_INITIAL_NUM_PLAYOUTS,
	       MCTS_NUM_PLAYOUTS_SCALE,
	       MCTS_DEFAULT_TIMEOUT,
	       std::move(playoutDriver)) {}
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
  const std::unique_ptr<PlayoutDriver> playoutDriver;
  std::shared_ptr<GameTree> tree;
  std::mutex treeMutex;
  std::thread workerThread;
  bool updateCancelled = false;
  bool running = false;

  void worker();
};

std::unique_ptr<Player> getPlayer(std::string name);
