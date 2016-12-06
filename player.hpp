#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <functional>

#define MCTS_DEFAULT_TIMEOUT 10 // Seconds
#define MCTS_DEFAULT_NUM_PLAYOUTS 200

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
  MCTSPlayer(unsigned numPlayouts,
	     unsigned timeout,
	     std::function<std::vector<PlayerId>(std::vector<State>)> playouts=devicePlayouts) :
    numPlayouts(numPlayouts),
    timeout(timeout),
    playouts(playouts)
  {}
  MCTSPlayer(std::function<std::vector<PlayerId>(std::vector<State>)> playouts=devicePlayouts) :
    MCTSPlayer(MCTS_DEFAULT_NUM_PLAYOUTS, MCTS_DEFAULT_TIMEOUT, playouts)
  {}
  ~MCTSPlayer() {};

  Move getMove(const State&) const;
  std::string getName() { return "mcts"; }

private:
  unsigned numPlayouts;
  unsigned timeout;
  std::function<std::vector<PlayerId>(std::vector<State>)> playouts;
};
