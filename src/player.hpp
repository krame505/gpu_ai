#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <string>
#include <functional>

#define MCTS_DEFAULT_TARGET_ITERATIONS 70
#define MCTS_DEFAULT_TIMEOUT 7 // Seconds
#define MCTS_INITIAL_NUM_PLAYOUTS 5000
#define MCTS_NUM_PLAYOUTS_SCALE 1.2
//#define MCTS_DEFAULT_DEVICE_HOST_PLAYOUT_RATIO 6
//#define MCTS_DEFAULT_PLAYOUT_DRIVER HybridPlayoutDriver(MCTS_DEFAULT_DEVICE_HOST_PLAYOUT_RATIO)
#define MCTS_DEFAULT_PLAYOUT_DRIVER DeviceCoarsePlayoutDriver


class Player {
public:
  virtual ~Player() {};

  virtual Move getMove(const State&) = 0;
  virtual std::string getName() const = 0;
};

class HumanPlayer : public Player {
public:
  ~HumanPlayer() {};

  Move getMove(const State&);
  std::string getName() const { return "human"; }
};

class RandomPlayer : public Player {
public:
  ~RandomPlayer() {};

  Move getMove(const State&);
  std::string getName() const { return "random"; }
};

class MCTSPlayer : public Player {
public:
  MCTSPlayer(unsigned numPlayouts,
	     unsigned targetIterations,
	     unsigned timeout,
	     PlayoutDriver *playoutDriver=new MCTS_DEFAULT_PLAYOUT_DRIVER) :
    numPlayouts(numPlayouts),
    targetIterations(targetIterations),
    timeout(timeout),
    playoutDriver(playoutDriver)
  {}
  MCTSPlayer(PlayoutDriver *playoutDriver=new MCTS_DEFAULT_PLAYOUT_DRIVER) :
    MCTSPlayer(MCTS_INITIAL_NUM_PLAYOUTS,
	       MCTS_DEFAULT_TARGET_ITERATIONS,
	       MCTS_DEFAULT_TIMEOUT,
	       playoutDriver)
  {}
  ~MCTSPlayer() {
    delete playoutDriver;
  };

  Move getMove(const State&);
  std::string getName() const { return "mcts"; }

private:
  unsigned numPlayouts;
  const unsigned targetIterations;
  const unsigned timeout;
  PlayoutDriver *playoutDriver;
};

Player *getPlayer(std::string name);
