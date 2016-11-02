
#include "state.hpp"

Player nextTurn(Player turn) {
  switch (turn) {
  case PLAYER_1:
    return PLAYER_2;
  case PLAYER_2:
    return PLAYER_1;
  }
}
