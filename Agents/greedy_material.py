import numpy as np
from Engine.state import State

COLORS = ["w", "b"]
MAX_DEPTH = 4


class Agent:
    def get_move(self, state):
        return self._get_move_helper(state, state.turn==1)[0]

    def _get_move_helper(self, state, maximizing=True, depth=0):
        if depth == MAX_DEPTH:
            return (None, self._count_material(state))
        best_move = None
        best_material = float('-inf') if maximizing else float('inf')
        for move in state.get_moves():
            curr_state = State(state.board_to_string() + '!')
            if curr_state.play_move(move):
                mat = self._count_material(curr_state)
            else:
                mat = self._get_move_helper(curr_state, (not maximizing), depth+1)[1]
            if (mat >= best_material if maximizing else mat <= best_material):
                best_material = mat
                best_move = move
        return (best_move, best_material)

    def _count_material(self, state):
        mat = 0
        for i in (-1, 1):
            color = COLORS[max(0, i)]
            mat += i * 9 * len(state.pieces[(color, "q")])
            mat += i * 5 * len(state.pieces[(color, "r")])
            mat += i * 3 * len(state.pieces[(color, "b")])
            mat += i * 3 * len(state.pieces[(color, "n")])
            mat += i * 1 * len(state.pieces[(color, "p")])
        return mat
