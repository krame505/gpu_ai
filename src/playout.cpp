
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
    while (!state.isGameOver()) {
      Move move = player.getMove(state);
      state.move(move);
    }
    results[i] = state.getNextTurn();
  }
  
  return results;
}

vector<PlayerId> HostFastPlayoutDriver::runPlayouts(vector<State> states) const {
  vector<PlayerId> results(states.size());

  #pragma omp parallel for
  for (unsigned i = 0; i < states.size(); i++) {
    State state = states[i];
    while (!state.isGameOver()) {
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
      for (unsigned i = 0; i < 42 && moves[state.turn].size() > 0; i++) {
	Move move = moves[state.turn][rand() % moves[state.turn].size()];
	for (const Move &prevMove : movesToApply) {
	  if (prevMove.conflictsWith(move))
	    goto end; // Sorry, but this is the best way to break out of a nested loop
	}
	movesToApply.push_back(move);
	state.turn = state.getNextTurn();
      }
    end:

      // Apply all the chosen moves
      for (const Move &move : movesToApply) {
	state.move(move);
      }

      // Other version of performing multiple moves that checks each move is valid with
      // isValidMove instead of checking them one-by-one
      /*PlayerId turn = state.turn;
      Move move = moves[turn][rand() % moves[turn].size()];
      do {
	state.move(move);
	turn = state.getNextTurn();
	if (moves[turn].size() == 0)
	  break;
	move = moves[turn][rand() % moves[turn].size()];
      } while (state.isValidMove(move));*/
    }
    results[i] = state.getNextTurn();
  }
  
  return results;
}
