
#include "playout.hpp"
#include "player.hpp"

#include <vector>
#include <random>
#include <cassert>
using namespace std;

vector<PlayerId> HostPlayoutDriver::runPlayouts(vector<State> states) const {
  vector<PlayerId> results(states.size());
  RandomPlayer player;

  #pragma omp parallel for
  for (unsigned i = 0; i < states.size(); i++) {
    State state = states[i];
    while (!state.isFinished()) {
      Move move = player.getMove(state);
      state.move(move);
    }
    results[i] = state.result();
  }
  
  return results;
}

vector<PlayerId> HostFastPlayoutDriver::runPlayouts(vector<State> states) const {
  vector<PlayerId> results(states.size());

  #pragma omp parallel for
  for (unsigned i = 0; i < states.size(); i++) {
    State state = states[i];
    while (!state.isFinished()) {
      // Generate all the moves possible for both players
      uint8_t numMoves[NUM_PLAYERS];
      Move result[NUM_PLAYERS][MAX_MOVES];
      bool genMovesForPlayer[NUM_PLAYERS] = {true, true};
      state.genMoves(numMoves, result, genMovesForPlayer);
      vector<Move> moves[NUM_PLAYERS] =
	{vector<Move>(result[PLAYER_1], result[PLAYER_1] + numMoves[PLAYER_1]),
	 vector<Move>(result[PLAYER_2], result[PLAYER_2] + numMoves[PLAYER_2])};

      // Choose random moves until one conflicting with a previous move is chosen
      vector<Move> movesToApply;
      PlayerId turn = state.turn;
      Move move = moves[turn][rand() % moves[turn].size()];
      bool valid = true;
      while (valid) {
	movesToApply.push_back(move);
	turn = nextTurn(turn);
	move = moves[turn][rand() % moves[turn].size()];
	for (const Move &prevMove : movesToApply) {
	  valid &= !prevMove.conflictsWith(move);
	}
      }

      // Apply all the chosen moves
      for (const Move &move : movesToApply) {
	state.move(move);
      }
    }
    results[i] = state.result();
  }
  
  return results;
}
