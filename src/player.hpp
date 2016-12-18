#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <functional>

#define MCTS_DEFAULT_TIMEOUT 7 // Seconds
#define MCTS_DEFAULT_NUM_PLAYOUTS 5000

class Player {
public:
  virtual ~Player() {};

  virtual Move getMove(const State&) const = 0;
  virtual std::string getName() const = 0;
};

class HumanPlayer : public Player {
public:
  ~HumanPlayer() {};

  Move getMove(const State&) const;
  std::string getName() const { return "human"; }
};

class RandomPlayer : public Player {
public:
  ~RandomPlayer() {};

  Move getMove(const State&) const;
  std::string getName() const { return "random"; }
};

class MCTSPlayer : public Player {
public:
  MCTSPlayer(unsigned numPlayouts,
	     unsigned timeout,
	     PlayoutDriver *playoutDriver=new DeviceSinglePlayoutDriver) :
    numPlayouts(numPlayouts),
    timeout(timeout),
    playoutDriver(playoutDriver)
  {}
  MCTSPlayer(PlayoutDriver *playoutDriver=new DeviceSinglePlayoutDriver) :
    MCTSPlayer(MCTS_DEFAULT_NUM_PLAYOUTS, MCTS_DEFAULT_TIMEOUT, playoutDriver)
  {}
  ~MCTSPlayer() {
    delete playoutDriver;
  };

  Move getMove(const State&) const;
  std::string getName() const { return "mcts"; }

private:
  unsigned numPlayouts;
  unsigned timeout;
  PlayoutDriver *playoutDriver;
};
