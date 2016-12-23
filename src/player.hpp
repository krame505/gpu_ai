#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <string>
#include <functional>

#define MCTS_INITIAL_NUM_PLAYOUTS 50
#define MCTS_FINAL_NUM_PLAYOUTS 600
#define MCTS_NUM_PLAYOUTS_SCALE 1.2
#define MCTS_DEFAULT_TARGET_ITERATIONS 900
#define MCTS_DEFAULT_TIMEOUT 10 // Seconds
#define MCTS_DEFAULT_PLAYOUT_DRIVER OptimalPlayoutDriver


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
  MCTSPlayer(unsigned initialNumPlayouts,
	     unsigned finalNumPlayouts,
	     unsigned targetIterations,
	     unsigned timeout,
	     PlayoutDriver *playoutDriver=new MCTS_DEFAULT_PLAYOUT_DRIVER) :
    initialNumPlayouts(initialNumPlayouts),
    finalNumPlayouts(finalNumPlayouts),
    targetIterations(targetIterations),
    timeout(timeout),
    playoutDriver(playoutDriver)
  {}
  MCTSPlayer(PlayoutDriver *playoutDriver=new MCTS_DEFAULT_PLAYOUT_DRIVER) :
    MCTSPlayer(MCTS_INITIAL_NUM_PLAYOUTS,
	       MCTS_FINAL_NUM_PLAYOUTS,
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
  unsigned initialNumPlayouts;
  unsigned finalNumPlayouts;
  const unsigned targetIterations;
  const unsigned timeout;
  PlayoutDriver *playoutDriver;
};

Player *getPlayer(std::string name);
