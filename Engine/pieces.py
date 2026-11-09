import numpy as np

BOARD_SIZE = 5

def _generate_moves(dirs, dist):
        ret = np.full((len(dirs), dist), fill_value=None)
        for row in range(len(dirs)):
            for i in range(dist):
                ret[row, i] = ((i+1)*dirs[row][0], (i+1)*dirs[row][1])
        return ret

# set up available moves
dirs = {}
dirs["k"] = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i==0 and j==0))
dirs["q"] = dirs["k"]
dirs["b"] = tuple((i, j) for i in (1, -1) for j in (1, -1))
dirs["n"] = tuple((k[0]*i, k[1]*j) for i in (1, -1) for j in (1, -1) for k in ((2, 1), (1, 2)))
dirs["r"] = tuple((0, i)[::j] for i in (-1, 1) for j in (-1, 1))
dirs["wp"] = tuple((-1, i) for i in (-1, 0, 1))
dirs["bp"] = tuple((1, i) for i in (-1, 0, 1))

MOVES = {}
recurse = ("q", "r", "b")
for piece in ["k", "q", "b", "n", "r", "wp", "bp"]:
    MOVES[piece] = _generate_moves(dirs[piece], BOARD_SIZE-1 if piece in recurse else 1)


class Piece:
    def __init__(self, color, piece_type, pos=None):
        assert color in ("w, b")
        assert pos is None or ((0 <= pos[0] < BOARD_SIZE) and (0 <= pos[1] < BOARD_SIZE))
        self.color = color
        self.type = piece_type
        self.pos = pos
        self.moves = ()

    def get_moves(self):
        return self.moves

    def play_move(self, move):
        assert move[0] == self.pos
        assert (0 <= move[1][0] < BOARD_SIZE) and (0 <= move[1][1] < BOARD_SIZE)
        self.pos = move[1]

    def equals(self, piece):
        return type(piece) == type(self) and piece.color == self.color and piece.pos == self.pos

    def get_piece_type(self):
        return (self.color, self.type)


class King(Piece):
    def __init__(self, color, pos=None):
        if not pos: pos = {"w": (4, 4), "b": (0, 0)}[color]
        super().__init__(color, "k", pos)
        self.moves = MOVES["k"]

class Queen(Piece):
    def __init__(self, color, pos=None):
        if not pos: pos = {"w": (4, 3), "b": (0, 1)}[color]
        super().__init__(color, "q", pos)
        self.moves = MOVES["q"]

class Bishop(Piece):
    def __init__(self, color, pos=None):
        if not pos: pos = {"w": (4, 2), "b": (0, 2)}[color]
        super().__init__(color, "b", pos)
        self.moves = MOVES["b"]

class Knight(Piece):
    def __init__(self, color, pos=None):
        if not pos: pos = {"w": (4, 1), "b": (0, 3)}[color]
        super().__init__(color, "n", pos)
        self.moves = MOVES["n"]

class Rook(Piece):
    def __init__(self, color, pos=None):
        if not pos: pos = {"w": (4, 0), "b": (0, 4)}[color]
        super().__init__(color, "r", pos)
        self.moves = MOVES["r"] 
    
class Pawn(Piece):
    def __init__(self, color, pos=None):
        if not pos: pos = {"w": (3, 0), "b": (1, 4)}[color]
        super().__init__(color, "p", pos)
        self.moves = MOVES[color + "p"]