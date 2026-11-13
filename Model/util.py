from Engine.state import BOARD_SIZE
import numpy as np

def _move_to_logit(move):
    logit = 0
    for i in range(2):
        for j in range(2):
            logit += move[i][j] * (BOARD_SIZE**((2*i)+j))
    return logit

def _logit_to_move(logit):
    move = []
    i = 0
    for p in range(3, -1, -1):
        bucket_size = BOARD_SIZE**p
        bucket = (logit - i) // bucket_size
        move.append(bucket)
        i += bucket * bucket_size
    return ((move[3], move[2]), (move[1], move[0]))


MOVES = (((a, b), (c, d)) for a in range(BOARD_SIZE) for b in range(BOARD_SIZE) for c in range(BOARD_SIZE) for d in range(BOARD_SIZE))
MOVE_TO_LOGIT = {move: _move_to_logit(move) for move in MOVES}
LOGIT_TO_MOVE = {logit: _logit_to_move(logit) for logit in range(BOARD_SIZE**4)}


def create_mask(logits):
    """
    Return a boolean mask given list of legal move logits
    """
    mask = np.zeros(BOARD_SIZE**4, dtype=bool)
    mask[logits] = True
    return mask


def _flip(move):    # for black's moves 
    return tuple((BOARD_SIZE - 1 - r, c) for r, c in move)

def encode(move, turn):    # absolute move --> network logit
    return MOVE_TO_LOGIT[_flip(move) if turn == 1 else move]

def decode(logit, turn):    # network logit -> absolute move
    move = LOGIT_TO_MOVE[logit]
    return _flip(move) if turn == 1 else move


def terminal_value(term):
    """ Get terminal value from State string"""
    if term is None:
        return None
    if term[0:1] == 'W':
        return 1
    if term[0:1] == 'B':
        return -1
    else:
        return 0  