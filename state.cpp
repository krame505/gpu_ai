// Contains the C++-only implementation of functions that only run on the host
// This file is compiled with gcc/g++

#include "state.hpp"
#include "colors.h"

#include <cassert>
using namespace std;

vector<Move> State::getMoves() const {
  uint8_t numMoves[NUM_PLAYERS];
  Move result[NUM_PLAYERS][MAX_MOVES];
  bool genMovesForPlayer[NUM_PLAYERS] = {false, false};
  genMovesForPlayer[turn] = true;
  genMoves(numMoves, result, genMovesForPlayer);

  return vector<Move>(result[turn], &result[turn][numMoves[turn]]);
}

bool State::isFinished() const {
  return getMoves().size() == 0;
}

ostream &operator<<(ostream &os, PlayerId pi) {
  switch (pi) {
  case PLAYER_1:
    return os << "Player 1";
  case PLAYER_2:
    return os << "Player 2";
  case PLAYER_NONE:
    return os << "Player NONE";
  default:
    assert(false);
    return os; // Unreachable, but to make -Wall shut up
  }
}

ostream &operator<<(ostream &os, PieceType pt) {
  switch (pt) {
  case PLAYER_1:
    return os << "checker";
  case PLAYER_2:
    return os << "king";
  default:
    assert(false);
    return os; // Unreachable, but to make -Wall shut up
  }
}

ostream &operator<<(ostream &os, Loc loc) {
  return os << string(1, 'a' + loc.col) << (BOARD_SIZE - loc.row);
}

ostream &operator<<(ostream &os, Move m) {
  os << m.from << " to " << m.to;
  if (m.jumps) {
    if (m.jumps > 1) {
      os << " via";
      for (int i = 0; i < m.jumps - 1; i++) {
        os << " " << m.intermediate[i];
      }
    }    
    os << ", captured";
    for (unsigned i = 0; i < m.jumps; i++) {
      os << " " << m.removed[i];
    }
  }
  if (m.promoted)
    os << ", promoted to king";
  return os;
}

ostream &operator<<(ostream &os, BoardItem bi) {
  if (!bi.occupied)
    return os << "  ";
  switch (bi.type) {
  case CHECKER:
    switch (bi.owner) {
#ifdef NOUNICODE
    case PLAYER_1:
      return os << "o";
    case PLAYER_2:
      return os << "O";
#else
    case PLAYER_1:
      return os << "⛂";
    case PLAYER_2:
      return os << "⛀";
#endif
    default:
      break;
    }
    break;

  case CHECKER_KING:
    switch (bi.owner) {
#ifdef NOUNICODE
    case PLAYER_1:
      return os << "k";
    case PLAYER_2:
      return os << "K";
#else
    case PLAYER_1:
      return os << "⛃";
    case PLAYER_2:
      return os << "⛁";
#endif
    default:
      break;
    }
  }
  
  // Didn't match, should never happen
  assert(false);
  return os;
}

ostream &operator<<(ostream &os, State s) {
  os << s.turn << "'s turn" << endl;
  for (int i = 0; i < BOARD_SIZE; i++) {
    os << "  " << string(1, 'a' + i);
  }
  os << "\n";
  for (int i = 0; i < BOARD_SIZE; i++) {
    os << (BOARD_SIZE - i) << " ";
    for (int j = 0; j < BOARD_SIZE; j++) {
      os << EFFECT(BACKGROUND(!((i + j) % 2)));
      if (s.board[i][j].occupied) {
        os << s.board[i][j] << " ";
      }
      else {
        os << EFFECT(FOREGROUND(BLUE));
        os << string(1, 'a' + j) << (BOARD_SIZE - i);
      }
      os << EFFECT(FOREGROUND(DEFAULT));
      os << EFFECT(BACKGROUND(DEFAULT));
      os << " ";
    }
    os << (BOARD_SIZE - i) << "\n";
  }
  for (int i = 0; i < BOARD_SIZE; i++) {
    os << "  " << string(1, 'a' + i);
  }
  return os;
}

void printLoc(const Loc &loc) {
  cout << loc << endl;
}

void printMove(const Move &move) {
  cout << move << endl;
}

void printState(const State &state) {
  cout << state << endl;
}
