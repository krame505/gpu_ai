#pragma once

#include "state.hpp"
#include "playout.hpp"

#include <array>
#include <vector>

class GameTree {
public:
  GameTree(State state, GameTree *const parent):
    state(state),
    parent(parent),
    trials(0) {
    for (Move m : state.moves()) {
      moves.push_back(m);
      children.push_back(NULL);
    }
  }

  ~GameTree() {
    for (GameTree *n : children)
      if (n != NULL)
        delete n;
  }

  std::vector<GameTree*> getChildren() const {
    return children;
  }

  // Compute the fraction of wins for each player
  std::array<double, NUM_PLAYERS> getScores() const;

  // Compute the Upper Confidence Bound 1 scoring algorithm
  double ucb1() const;

  // Distrubute trials to perform based on UCB1, creating nodes as needed
  std::vector<State> select(unsigned trials);
  
  // Update the entire tree with the playout results
  void update(std::vector<Player>);
  
private:
  const State state;
  const GameTree *parent;
  std::vector<Move> moves;
  std::vector<GameTree*> children;

  unsigned trials; // Trials assigned in last pass
  unsigned totalTrials;
  unsigned wins[NUM_PLAYERS] = {0};
};

// Build a tree by performing a series of trials with the number of playouts in a vector
GameTree *buildTree(State, std::vector<unsigned> trials);
