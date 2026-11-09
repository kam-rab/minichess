import numpy as np

class Agent:
    def get_move(self, state):
        moves = state.get_moves()
        return moves[np.random.choice(range(len(moves)))]
