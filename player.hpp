#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <functional>

#define MCTS_DEFAULT_NUM_ITERATIONS 8
#define MCTS_DEFAULT_NUM_TRIALS 100

class Player {
public:
  virtual ~Player() {};

  virtual Move getMove(const State&) const = 0;
  virtual std::string getName() = 0;
};

class HumanPlayer : public Player {
public:
  ~HumanPlayer() {};

  Move getMove(const State&) const;
  std::string getName() { return "human"; }
};

class RandomPlayer : public Player {
public:
  ~RandomPlayer() {};

  Move getMove(const State&) const;
  std::string getName() { return "random"; }
};

class MCTSPlayer : public Player {
public:
  MCTSPlayer(unsigned numIterations,
	     unsigned numTrials,
	     std::function<std::vector<PlayerId>(std::vector<State>)> playouts=devicePlayouts) :
    numIterations(numIterations),
    numTrials(numTrials),
    playouts(playouts)
  {}
  MCTSPlayer(std::function<std::vector<PlayerId>(std::vector<State>)> playouts=devicePlayouts) :
    MCTSPlayer(MCTS_DEFAULT_NUM_ITERATIONS, MCTS_DEFAULT_NUM_TRIALS, playouts)
  {}
  ~MCTSPlayer() {};

  Move getMove(const State&) const;
  std::string getName() { return "mcts"; }

private:
  unsigned numIterations;
  unsigned numTrials;
  std::function<std::vector<PlayerId>(std::vector<State>)> playouts;
};
