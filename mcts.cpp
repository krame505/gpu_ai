
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <cassert>
#include <cmath>
using namespace std;

vector<State> GameTree::select(unsigned trials) {
  // If this is a terminal state, then all assigned playouts must happen from this state
  if (state.isFinished()) {
    assignedTrials = trials;
    return vector<State>(trials, state);
  }

  // If this is an unexpanded leaf node on the tree
  if (!expanded) {
    // If there is more than 1 trial to allocate then expand this node and allocate trials to the new children
    if (trials > 1) {
      for (unsigned i = 0; i < children.size(); i++) {
	if (children[i] == NULL) {
	  State newState = state;
	  newState.move(moves[i]);
	  children[i] = new GameTree(newState, this);
	}
      }
      expanded = true;
    }
    // Otherwise if there is only one trial, perform a playout from this node
    else if (trials == 1) {
      assignedTrials = 1;
      return vector<State> {state};
    }
    // If there are no trials assigned return the empty vector
    else {
      assignedTrials = 0;
      return vector<State> {};
    }
  }

  unsigned childAssignedTrials[children.size()] = {0};
  assignedTrials = 0;

  // Calculate weights for all children
  double weights[children.size()];
  double totalWeights = 0;
  unsigned numUntried = 0;
  for (unsigned i = 0; i < children.size(); i++) {
    weights[i] = children[i]->ucb1();
    totalWeights += weights[i];
    if (children[i]->totalTrials == 0)
      numUntried++;
  }
  
  // Assign trials to children based on weights
  for (unsigned i = 0; i < children.size(); i++) {
    // If some children are untried, evenly assign all trials to them
    if (numUntried > 0) {
      if (children[i]->totalTrials == 0)
	childAssignedTrials[i] = trials / numUntried;
      else
	childAssignedTrials[i] = 0;
    }
    // Assign trials based on fractions of weights, rounding down
    else if (trials > 0) {
      childAssignedTrials[i] = trials * (weights[i] / totalWeights);
    }
    assignedTrials += childAssignedTrials[i];
  }

  // Assign extra trials in sorted order of weights
  // Kinda inefficent but number of children is small
  bool assigned[children.size()] = {false};
  while (assignedTrials < trials) {
    double maxWeight = -INFINITY;
    int opt = -1;
    for (unsigned i = 0; i < children.size(); i++) {
      if (!assigned[i] && weights[i] > maxWeight) {
	maxWeight = weights[i];
	opt = i;
      }
    }
    assert(opt != -1);      
    assigned[opt] = true;
    childAssignedTrials[opt]++;
    assignedTrials++;
  }

  assert(assignedTrials == trials);
  
  // Recursively assign trials to children
  vector<State> result;
  for (unsigned i = 0; i < children.size(); i++) {
    vector<State> childTrials = children[i]->select(childAssignedTrials[i]);
    result.insert(result.end(), childTrials.begin(), childTrials.end());
  }

  return result;
}

void GameTree::update(const vector<PlayerId> &results) {
  assert(results.size() == assignedTrials);

  totalTrials += assignedTrials;
  if (expanded) {
    // Maintain an iterator to the input results vector
    auto it = results.begin();
    for (GameTree *child : children) {
      // Copy the next child->assignedTrials results into childResults
      vector<PlayerId> childResults(it, it + child->assignedTrials);
      it += child->assignedTrials;

      // Recursively perform an update
      child->update(childResults);
	
      // Update the current wins with the child's wins
      for (unsigned i = 0; i < NUM_PLAYERS; i++) {
	wins[i] += child->wins[i];
      }
    }
  }
  else {
    for (PlayerId result : results)
      wins[result]++;
  }
}

double GameTree::ucb1() const {
  assert(parent != NULL);
  // Weight untried nodes the highest
  if (totalTrials == 0)
    return INFINITY;
  // Weight everything else by (fraction of wins) + (untried exploration factor)
  else
    return (double)wins[state.turn] / totalTrials +
      sqrt(2.0L * log(parent->totalTrials) / totalTrials);
}

double GameTree::getScore(PlayerId player) const {
  if (state.isFinished()) {
    PlayerId result = state.result();
    if (result == player)
      return 1;
    else
      return 0;
  }
  else
    return (double)wins[player] / totalTrials;
}

Move GameTree::getOptMove(PlayerId player) const {
  double highestScore = -INFINITY;
  Move bestMove;
    
  for (unsigned i = 0; i < children.size(); i++) {
    if (children[i]->getScore(player) > highestScore) {
      highestScore = children[i]->getScore(player);
      bestMove = moves[i];
    }
  }
 
  assert(highestScore != -INFINITY);
  
  return bestMove;
}

GameTree *buildTree(State state, const vector<unsigned> &trials) {
  GameTree *tree = new GameTree(state, NULL);
  for (unsigned numPlayouts : trials) {
    vector<State> playoutStates = tree->select(numPlayouts);
    vector<PlayerId> results = playouts(playoutStates);
    tree->update(results);
  }
  return tree;
}
