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

  virtual std::string getName() const = 0;
  virtual Move getMove(const State&, bool verbose=true) = 0;
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

  std::string getName() const { return "mcts"; }
  Move getMove(const State&, bool verbose=true);

private:
  const unsigned initialNumPlayouts;
  const float numPlayoutsScale;
  const unsigned timeout;
  PlayoutDriver *playoutDriver;
};

Player *getPlayer(std::string name);
