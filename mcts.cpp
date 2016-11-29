
#include "mcts.hpp"
#include "playout.hpp"

#include <vector>
#include <cassert>
#include <cmath>
using namespace std;

vector<State> GameTree::select(unsigned trials) {
  assert(!state.isFinished());

  // Create new children if any are null
  vector<unsigned> newNodes;
  for (unsigned i = 0; i < children.size() && newNodes.size() < trials; i++) {
    if (children[i] == NULL) {
      State newState = state;
      newState.move(moves[i]);
      children[i] = new GameTree(newState, this);
      newNodes.push_back(i);
    }
  }

  unsigned childAssignedTrials[children.size()] = {0};
  assignedTrials = 0;

  // If there are new children created, assign all trials evenly between them
  if (newNodes.size() > 0) {
    for (unsigned newNode : newNodes) {
      childAssignedTrials[newNode] = (double)trials / newNodes.size();
      assignedTrials += childAssignedTrials[newNode];
    }
    
    for (unsigned newNode : newNodes) {
      if (assignedTrials >= trials)
        break;
      childAssignedTrials[newNode]++;
      assignedTrials++;
    }
  }
  // If there are no new children, assign trials based on UCB1 scoring
  else {
    double weights[children.size()];
    double totalWeights = 0;

    for (unsigned i = 0; i < children.size(); i++) {
      weights[i] = children[i]->ucb1();
      totalWeights += weights[i];
    }

    for (unsigned i = 0; i < children.size(); i++) {
      childAssignedTrials[i] = trials * (weights[i] / totalWeights);
      assignedTrials += childAssignedTrials[i];
    }

    for (unsigned i = 0; i < trials - assignedTrials; i++) {
      childAssignedTrials[i]++;
      assignedTrials++;
    }
  }

  assert(assignedTrials == trials);
  
  vector<State> result;
  for (unsigned i = 0; i < children.size(); i++) {
    if (childAssignedTrials[i] > 1) {
      vector<State> childTrials = children[i]->select(childAssignedTrials[i]);
      result.insert(result.end(), childTrials.begin(), childTrials.end());
    }
    else if (childAssignedTrials[i] == 1) {
      children[i]->assignedTrials = 1;
      result.push_back(state);
    }
  }
  return result;
}

void GameTree::update(const vector<PlayerId> &results) {
  totalTrials += assignedTrials;
  auto it = results.begin();
  for (GameTree *child : children) {
    if (child != NULL) {
      vector<PlayerId> childResults(it, it += child->assignedTrials);
      child->update(childResults);
      for (unsigned i = 0; i < NUM_PLAYERS; i++) {
	wins[i] += child->wins[i];
      }
    }
  }
}

double GameTree::getScore(PlayerId player) const {
  return (double)wins[player] / totalTrials;
}

double GameTree::ucb1() const {
  assert(parent != NULL);
  if (totalTrials == 0)
    return INFINITY;
  else
    return (double)wins[state.turn] / totalTrials +
      sqrt(2.0L * log(parent->totalTrials) / totalTrials);
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
