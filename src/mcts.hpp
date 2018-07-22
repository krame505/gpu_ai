#pragma once

#include "state.hpp"
#include "playout.hpp"

#include <vector>
#include <mutex>

#include <unistd.h>

class GameTree {
public:
  GameTree(State state) : GameTree(state, NULL) {}
  ~GameTree() {
    if (expanded)
      for (GameTree *n : children)
        delete n;
  }

  // Perform a move, selecting a branch of the tree and discarding the rest
  GameTree *move(const Move &move);

  // Compute the fraction of wins for a player
  double getScore(PlayerId player) const;

  // Compute the best move for a player based on scores for children
  Move getOptMove(PlayerId player) const;

  // Compute the best move for a player based on scores for children
  unsigned getTotalTrials() const {
    return totalTrials;
  }

  // Expand a tree by selecting playouts, performing them, and updating the tree
  void expand(unsigned numPlayouts, PlayoutDriver *playoutDriver);
  
  // Distrubute trials to perform based on UCB1, creating nodes as needed
  std::vector<State> select(unsigned trials);
  
  // Update the entire tree with the playout results
  void update(const std::vector<PlayerId>&);
  
  const State state;

private:
  GameTree(State state, GameTree *const parent) :
    state(state),
    parent(parent) {
    for (Move &m : state.getMoves()) {
      moves.push_back(m);
      children.push_back(NULL);
    }
  }

  // Compute the Upper Confidence Bound 1 scoring algorithm
  double ucb1() const;

  const GameTree *parent;
  bool expanded = false;
  std::vector<Move> moves;
  std::vector<GameTree*> children;

  unsigned assignedTrials = 0; // Trials assigned in last pass
  unsigned totalTrials    = 0; // Total (finished) trials from this tree
  unsigned wins[NUM_PLAYERS] = {0};
};

