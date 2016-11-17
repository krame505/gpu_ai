// Contains the C++-only implementation of functions that only run on the host
// This file is compiled with gcc/g++

#include "state.hpp"
#include "colors.h"

#include <cassert>
using namespace std;

PlayerId State::result() const {
  int numPieces[NUM_PLAYERS] = {0, 0};
  int numKings[NUM_PLAYERS]  = {0, 0};

  for (int i = 0; i < BOARD_SIZE; i++) {
    for (int j = 0; j < BOARD_SIZE; j++) {
      if (board[i][j].occupied) {
	numPieces[board[i][j].owner]++;
	if (board[i][j].type == CHECKER_KING)
	  numKings[board[i][j].owner]++;
      }
    }
  }
  
  if (numPieces[PLAYER_1] == numPieces[PLAYER_2]) {
    if (numKings[PLAYER_1] == numKings[PLAYER_2])
      return PLAYER_NONE;
    else
      return numKings[PLAYER_1] > numKings[PLAYER_2] ? PLAYER_1 : PLAYER_2;
  } else {
    return numPieces[PLAYER_1] > numPieces[PLAYER_2] ? PLAYER_1 : PLAYER_2;
  }
}


bool State::isFinished() const {
  return moves().size() == 0;
  //return false; // TODO
}

vector<Move> State::moves() const {
  vector<Move> result;
  Move moves[MAX_LOC_MOVES];
  for (uint8_t row = 0; row < BOARD_SIZE; row++) {
    for (uint8_t col = 0; col < BOARD_SIZE; col++) {
        //TODO: fix locMoves so that only the moves for the current player are
        //considered
      uint8_t numMoves = locMoves(Loc(row, col), moves);
      result.insert(result.end(), &moves[0], &moves[numMoves]);
    }
  }
  return result;
}

ostream &operator<<(ostream &os, PlayerId pi) {
  switch (pi) {
  case PLAYER_1:
    return os << "Player 1";
  case PLAYER_2:
    return os << "Player 2";
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
  return os << string(1, 'a' + loc.col) << (loc.row + 1);
}

ostream &operator<<(ostream &os, Move m) {
  os << m.from << " to " << m.to;
  if (m.jumps) {
    os << ", captured";
    for (unsigned i = 0; i < m.jumps; i++) {
      os << " " << m.removed[i];
    }
  }
  if (m.promoted)
    os << ", promoted to " << m.newType;
  return os;
}

ostream &operator<<(ostream &os, BoardItem bi) {
  if (!bi.occupied)
    return os << "  ";
  switch (bi.type) {
  case CHECKER:
    switch (bi.owner) {
    case PLAYER_1:
      return os << "⛂";
    case PLAYER_2:
      return os << "⛀";
    default:
      break;
    }
    break;

  case CHECKER_KING:
    switch (bi.owner) {
    case PLAYER_1:
      return os << "⛃";
    case PLAYER_2:
      return os << "⛁";
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
      // TODO: check parity of background colors is correct
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
