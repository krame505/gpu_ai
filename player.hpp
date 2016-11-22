#pragma once

#include "state.hpp"
#include "mcts.hpp"

#include <vector>

class Player {
public:
  virtual ~Player() {};

  virtual Move getMove(const State&) const = 0;
  virtual void PrintType() { std::cout << "none"; }
};

class HumanPlayer : public Player {
public:
  ~HumanPlayer() {};
  Move getMove(const State&) const;
  void PrintType() { std::cout << "human"; }
};

class RandomPlayer : public Player {
public:
  ~RandomPlayer() {};
  Move getMove(const State&) const;
  void PrintType() { std::cout << "random"; }
};

class MCTSPlayer : public Player {
public:
  ~MCTSPlayer() {};
  Move getMove(const State&) const;
  void PrintType() { std::cout << "mcts"; }
};
