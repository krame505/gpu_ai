#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <string>
#include <functional>

#define MCTS_INITIAL_NUM_PLAYOUTS 50
#define MCTS_NUM_PLAYOUTS_SCALE 1.005
#define MCTS_DEFAULT_TIMEOUT 10 // Seconds
#define MCTS_DEFAULT_PLAYOUT_DRIVER OptimalPlayoutDriver


class Player {
public:
  virtual ~Player() {};

  virtual Move getMove(const State&, bool verbose=true) = 0;
  virtual std::string getName() const = 0;
};

class HumanPlayer : public Player {
public:
  ~HumanPlayer() {};

  Move getMove(const State&, bool verbose=true);
  std::string getName() const { return "human"; }
};

class RandomPlayer : public Player {
public:
  ~RandomPlayer() {};

  Move getMove(const State&, bool verbose=true);
  std::string getName() const { return "random"; }
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
    playoutDriver(playoutDriver)
  {}
  MCTSPlayer(PlayoutDriver *playoutDriver=new MCTS_DEFAULT_PLAYOUT_DRIVER) :
    MCTSPlayer(MCTS_INITIAL_NUM_PLAYOUTS,
	       MCTS_NUM_PLAYOUTS_SCALE,
	       MCTS_DEFAULT_TIMEOUT,
	       playoutDriver)
  {}
  ~MCTSPlayer() {
    delete playoutDriver;
  };

  Move getMove(const State&, bool verbose=true);
  std::string getName() const { return "mcts"; }

private:
  const unsigned initialNumPlayouts;
  const float numPlayoutsScale;
  const unsigned timeout;
  PlayoutDriver *playoutDriver;
};

Player *getPlayer(std::string name);
