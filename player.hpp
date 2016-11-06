#pragma once

#include "state.hpp"
#include "mcts.hpp"

#include <vector>

class Player {
public:
  virtual ~Player() {};

  virtual Move getMove(const State&) const = 0;
};

class HumanPlayer : public Player {
public:
  ~HumanPlayer() {};
  Move getMove(const State&) const;
};

class RandomPlayer : public Player {
public:
  ~RandomPlayer() {};
  Move getMove(const State&) const;
};

class MCTSPlayer : public Player {
public:
  ~MCTSPlayer() {};
  Move getMove(const State&) const;
};
