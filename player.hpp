#pragma once

#include "state.hpp"
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <functional>

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
  MCTSPlayer(std::function<std::vector<PlayerId>(std::vector<State>)> playouts=devicePlayouts) :
    playouts(playouts)
  {}
  ~MCTSPlayer() {};

  Move getMove(const State&) const;
  std::string getName() { return "mcts"; }

private:
  std::function<std::vector<PlayerId>(std::vector<State>)> playouts;
};
