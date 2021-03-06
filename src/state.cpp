// Contains the C++-only implementation of functions that only run on the host
// This file is compiled with gcc/g++

#include "state.hpp"
#include "colors.h"

#include <cassert>
using namespace std;

vector<Move> State::getMoves() const {
  Move result[MAX_MOVES];
  uint8_t numMoves = genMoves(result);
  return vector<Move>(result, result + numMoves);
}

bool State::isGameOver() const {
  return getMoves().size() == 0 || movesSinceLastCapture >= NUM_DRAW_MOVES;
}

PlayerId State::getWinner() const {
  assert(isGameOver());
  return movesSinceLastCapture < NUM_DRAW_MOVES? getNextTurn() : PLAYER_NONE;
}

State getStartingState() {
  // Unspecified BoardItems are initialized to 0
  State state = {
    {{{}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}},
     {{true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}},
     {{}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}, {}, {true, CHECKER, PLAYER_1}},
     {{}, {}, {}, {}, {}, {}, {}, {}},
     {{}, {}, {}, {}, {}, {}, {}, {}},
     {{true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}},
     {{}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}},
     {{true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}, {true, CHECKER, PLAYER_2}, {}}},
    PLAYER_1,
    0
  };
  return state;
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
    throw runtime_error("Unknown player id");
  }
}

ostream &operator<<(ostream &os, PieceType pt) {
  switch (pt) {
  case PLAYER_1:
    return os << "checker";
  case PLAYER_2:
    return os << "king";
  default:
    throw runtime_error("Unknown piece type");
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
  throw runtime_error("Unknown board item");
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
