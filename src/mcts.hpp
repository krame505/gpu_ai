#pragma once

#include "state.hpp"
#include "playout.hpp"

#include <vector>
#include <functional>

class GameTree {
public:
  GameTree(State state, GameTree *const parent=NULL) :
    state(state),
    parent(parent) {
    for (Move &m : state.getMoves()) {
      moves.push_back(m);
      children.push_back(NULL);
    }
  }

  ~GameTree() {
    if (expanded)
      for (GameTree *n : children)
	delete n;
  }

  // Distrubute trials to perform based on UCB1, creating nodes as needed
  std::vector<State> select(unsigned trials);
  
  // Update the entire tree with the playout results
  void update(const std::vector<PlayerId>&);

  // Compute the Upper Confidence Bound 1 scoring algorithm
  double ucb1() const;

  // Compute the fraction of wins for a player
  double getScore(PlayerId player) const;

  // Compute the best move for a player based scores for children
  Move getOptMove(PlayerId player) const;
  
  const State state;
private:
  const GameTree *parent;
  bool expanded = false;
  std::vector<Move> moves;
  std::vector<GameTree*> children;

  unsigned assignedTrials = 0; // Trials assigned in last pass
  unsigned totalTrials    = 0; // Total (finished) trials from this tree
  unsigned wins[NUM_PLAYERS] = {0};
};

// Build a tree by performing a series of trials with the number of playouts in a vector
GameTree *buildTree(State, const std::vector<unsigned> &trials,
		    std::function<std::vector<PlayerId>(std::vector<State>)> playouts);

// Build a tree by performing a series of trials until a timeout
GameTree *buildTree(State, unsigned numPlayouts, unsigned timeout,
		    std::function<std::vector<PlayerId>(std::vector<State>)> playouts);
