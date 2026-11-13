import numpy as np
from Engine.state import State
from Model.util import terminal_value

COLORS = ["w", "b"]
MAX_DEPTH = 4

NUM_PIECES_FOR_CHECK_MATE = 2   # tells the greedy agent to go an extra ply to search for mates if check_mate=True and if pieces <= NUM_PIECES_FOR_CHECK_MATE
MOVE_NUMBER_FOR_CHECK_MATE = 134  # tells the greedy agent to go an extra ply, same as above


class Agent:
    def get_move(self, state, check_mate=True):
        return self._get_move_helper(state, state.turn==0, check_mate=check_mate)[0]

    def _get_move_helper(self, state, maximizing=True, depth=0, alpha=float('-inf'), beta=float('inf'), check_mate=False):
        if depth == MAX_DEPTH:
            if not check_mate:
                return (None, self._count_material(state))
            else:
                mat, num_pieces = self._count_material(state, return_pieces=maximizing)
                if num_pieces > NUM_PIECES_FOR_CHECK_MATE or state.move_number < MOVE_NUMBER_FOR_CHECK_MATE:
                    return (None, mat)
                else:
                    for move in state.get_moves():
                        curr_state = State(state.state_to_string())
                        term = curr_state.play_move(move)
                        if term is None:
                            continue
                        sign = 1 if maximizing else -1
                        term = terminal_value(term) * sign
                        if term == 1:
                            return (None, sign* (1000 - depth))
                        if term == 0:
                            mat = max(mat, 0) if maximizing else min(mat, 0)
                    return (None, mat)

        best_move = None
        if maximizing:
            best_material = float('-inf')
            for move in state.get_moves():
                curr_state = State(state.state_to_string())
                term = curr_state.play_move(move)
                if term:
                    mat = terminal_value(term) * (1000 - depth)
                else:
                    mat = self._get_move_helper(curr_state, (not maximizing), depth+1, alpha, beta, check_mate=check_mate)[1]
                alpha = max(mat, alpha)
                if alpha >= beta:
                    return (None, alpha)
                if (mat > best_material):
                    best_material = mat
                    best_move = move
        else: 
            best_material = float('inf')
            for move in state.get_moves():
                curr_state = State(state.state_to_string())
                term = curr_state.play_move(move)
                if term:
                    mat = terminal_value(term) * (1000 - depth)
                else:
                    mat = self._get_move_helper(curr_state, (not maximizing), depth+1, alpha, beta, check_mate=check_mate)[1]
                beta = min(mat, beta)
                if alpha >= beta:
                    return (None, beta)
                if (mat < best_material):
                    best_material = mat
                    best_move = move
        
        return (best_move, best_material)


    def _count_material(self, state, return_pieces=None):
        mat = 0
        for i in (-1, 1):
            color = COLORS[max(0, i)]
            mat += (-i) * 9 * len(state.pieces[(color, "q")])
            mat += (-i) * 5 * len(state.pieces[(color, "r")])
            mat += (-i) * 3 * len(state.pieces[(color, "b")])
            mat += (-i) * 3 * len(state.pieces[(color, "n")])
            mat += (-i) * 1 * len(state.pieces[(color, "p")])
        if return_pieces is None:
            return mat

        color = "w" if return_pieces else "b"
        num_pieces = 0
        for piece in ["q", "r", "b", "n", "p"]:
            num_pieces += len(state.pieces[(color, piece)])
        return (mat, num_pieces)
