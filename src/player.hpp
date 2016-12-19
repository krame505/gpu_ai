#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <string>
#include <functional>

#define MCTS_DEFAULT_TIMEOUT 7 // Seconds
#define MCTS_DEFAULT_NUM_PLAYOUTS 50000
#define MCTS_DEFAULT_DEVICE_HOST_PLAYOUT_RATIO 6
#define MCTS_DEFAULT_PLAYOUT_DRIVER HybridPlayoutDriver(MCTS_DEFAULT_DEVICE_HOST_PLAYOUT_RATIO)


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
	     PlayoutDriver *playoutDriver=new MCTS_DEFAULT_PLAYOUT_DRIVER) :
    numPlayouts(numPlayouts),
    timeout(timeout),
    playoutDriver(playoutDriver)
  {}
  MCTSPlayer(PlayoutDriver *playoutDriver=new MCTS_DEFAULT_PLAYOUT_DRIVER) :
    MCTSPlayer(MCTS_DEFAULT_NUM_PLAYOUTS, MCTS_DEFAULT_TIMEOUT, playoutDriver)
  {}
  ~MCTSPlayer() {
    delete playoutDriver;
  };

  Move getMove(const State&) const;
  std::string getName() const { return "mcts"; }

private:
  const unsigned numPlayouts;
  const unsigned timeout;
  PlayoutDriver *playoutDriver;
};

Player *getPlayer(std::string name);
