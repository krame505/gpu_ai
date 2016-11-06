
#include "mcts.hpp"
#include "playout.hpp"

#include <array>
#include <vector>
#include <cassert>
#include <cmath>
using namespace std;

vector<State> GameTree::select(unsigned trials) {
  assert(!state.isFinished());
  this->trials = trials;

  vector<unsigned> newNodes;
  for (unsigned i = 0; i < children.size() && newNodes.size() < trials; i++) {
    if (children[i] == NULL) {
      State newState = state;
      newState.move(moves[i]);
      children[i] = new GameTree(newState, this);
      newNodes.push_back(i);
    }
  }

  unsigned assignedTrials[children.size()] = {0};
  unsigned totalAssignedTrials = 0;

  // If there are new children created, assign all trials evenly between them
  if (newNodes.size() > 0) {
    for (unsigned newNode : newNodes) {
      assignedTrials[newNode] = (double)trials / newNodes.size();
      totalAssignedTrials += assignedTrials[newNode];
    }
    
    for (unsigned newNode : newNodes) {
      if (totalAssignedTrials >= trials)
        break;
      assignedTrials[newNode]++;
    }
    
    assert(totalAssignedTrials == trials);
  
    vector<State> result;
    for (unsigned i = 0; i < children.size(); i++) {
      children[i]->trials = assignedTrials[i];
      for (unsigned j = 0; j < assignedTrials[i]; j++) {
        result.push_back(state);
      }
    }
    return result;
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
      assignedTrials[i] = trials * (weights[i] / totalWeights);
      totalAssignedTrials += assignedTrials[i];
    }

    for (unsigned i = 0; i < trials - totalAssignedTrials; i++)
      assignedTrials[i]++;
    
    assert(totalAssignedTrials == trials);
  
    vector<State> result;
    for (unsigned i = 0; i < children.size(); i++) {
      if (assignedTrials[i] != 0) {
        vector<State> childTrials = children[i]->select(assignedTrials[i]);
        result.insert(result.end(), childTrials.begin(), childTrials.end());
      }
    }
    return result;
  }
}

void GameTree::update(vector<PlayerId> results) {
  totalTrials += trials;
  auto it = results.begin();
  for (unsigned i = 0; i < children.size(); i++) {
    vector<PlayerId> childResults(it, it += children[i]->trials);
    children[i]->update(childResults);
    for (unsigned j = 0; j < NUM_PLAYERS; j++) {
      wins[j] += children[i]->wins[j];
    }
  }
}

array<double, NUM_PLAYERS> GameTree::getScores() const {
  array<double, NUM_PLAYERS> result;
  for (unsigned i = 0; i < NUM_PLAYERS; i++) {
    result[i] = (double)wins[i] / totalTrials;
  }
  return result;
}

double GameTree::ucb1() const {
  assert(parent != NULL);
  if (totalTrials == 0)
    return INFINITY;
  else
    return (double)wins[state.turn] / totalTrials +
      sqrt(2.0L * log(parent->trials) / totalTrials);
}

GameTree *buildTree(State state, vector<unsigned> trials) {
  GameTree *tree = new GameTree(state, NULL);
  for (unsigned numPlayouts : trials) {
    vector<State> playoutStates = tree->select(numPlayouts);
    vector<PlayerId> results = playouts(playoutStates);
    tree->update(results);
  }
  return tree;
}
