#pragma once

#include "state.hpp"
#include "playout.hpp"

#include <vector>

class GameTree {
public:
  GameTree(State state, GameTree *const parent) :
    state(state),
    parent(parent) {
    for (Move m : state.getMoves()) {
      moves.push_back(m);
      children.push_back(NULL);
    }
    for (unsigned i = 0; i < NUM_PLAYERS; i++) {
      wins[i] = 0;
    }
  }

  ~GameTree() {
    for (GameTree *n : children)
      if (n != NULL)
        delete n;
  }

  std::vector<Move> getMoves() const {
    return moves;
  }

  std::vector<GameTree*> getChildren() const {
    return children;
  }

  // Compute the fraction of wins for a player
  double getScore(PlayerId player) const;

  // Compute the Upper Confidence Bound 1 scoring algorithm
  double ucb1() const;

  // Distrubute trials to perform based on UCB1, creating nodes as needed
  std::vector<State> select(unsigned trials);
  
  // Update the entire tree with the playout results
  void update(const std::vector<PlayerId>&);
  
private:
  const State state;
  const GameTree *parent;
  std::vector<Move> moves;
  std::vector<GameTree*> children;

  unsigned assignedTrials = 0; // Trials assigned in last pass
  unsigned totalTrials    = 0; // Total (finished) trials from this tree
  unsigned wins[NUM_PLAYERS];
};

// Build a tree by performing a series of trials with the number of playouts in a vector
GameTree *buildTree(State, const std::vector<unsigned> &trials);
