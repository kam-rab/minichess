# Hold a board state, generate all legal moves, find terminal games
# positions are indexed (y, x) starting from top left

import numpy as np
from Engine.pieces import *

# initial board state (string format)
PIECE_TYPES = (('w', 'k'), ('w', 'q'), ('w', 'b'), ('w', 'n'), ('w', 'r'), ('w', 'p'), ('b', 'k'), ('b', 'q'), ('b', 'b'), ('b', 'n'), ('b', 'r'), ('b', 'p'))
COLORS = ('w', 'b')
STATE_INIT = "0!wbb(0, 2)bk(0, 0)bn(0, 3)bp(1, 0)(1, 1)(1, 2)(1, 3)(1, 4)bq(0, 1)br(0, 4)wb(4, 2)wk(4, 4)wn(4, 1)wp(3, 0)(3, 1)(3, 2)(3, 3)(3, 4)wq(4, 3)wr(4, 0)!1wbb(0, 2)bk(0, 0)bn(0, 3)bp(1, 0)(1, 1)(1, 2)(1, 3)(1, 4)bq(0, 1)br(0, 4)wb(4, 2)wk(4, 4)wn(4, 1)wp(3, 0)(3, 1)(3, 2)(3, 3)(3, 4)wq(4, 3)wr(4, 0)"


class State:
    def __init__(self, state=STATE_INIT):
        self.from_string(state)


    def _create_board(self, pieces):
        ret = np.full((BOARD_SIZE, BOARD_SIZE), fill_value=None)
        for piece in pieces:
            ret[piece.pos] = piece
        return ret

    
    def create_onehot_board(self):
        ret = np.empty((300,))
        i = 0
        for piece_type in self._get_piece_types():
            ret[i:i+25] = np.reshape(self.create_board(self.pieces[piece_type]), (-1, ))
            i += 25
        return ret


    def get_board(self):
        return self.board


    def _get_piece_types(self):
        piece_types = list(self.pieces.keys())
        piece_types.sort()
        return piece_types


    def get_moves(self, piece=None):
        moves = self._get_moves_before_illegal(self.turn, piece)
        for i in range(len(moves)-1, -1, -1):
            captured = self.play_move(moves[i], lookahead=True)  # never change self.pieces while looking ahead, as it never affects king being in check.
            rev = moves[i][::-1]   # remember move so it can be undone
            if self._in_check():
                moves = moves[:i] + moves[i+1:]
                i -= 1
            self.play_move(rev, lookahead=True)
            if captured:
                self.board[rev[0]] = captured  # return piece if captured
        return moves

    def _has_moves(self):
        has_moves = False
        for piece_type in self._get_piece_types()[(1-self.turn)*6 : (2-self.turn)*6][::-1]:
            for piece in self.pieces[piece_type]:
                moves = self._get_moves_before_illegal(self.turn, piece)
                for i in range(len(moves)-1, -1, -1):
                    captured = self.play_move(moves[i], lookahead=True)  # never change self.pieces while looking ahead, as it never affects king being in check.
                    rev = moves[i][::-1]   # remember move so it can be undone
                    if not self._in_check():
                        has_moves = True
                    self.play_move(rev, lookahead=True)
                    if captured:
                        self.board[rev[0]] = captured  # return piece if captured
                    if has_moves: 
                        return True
        return False


    def _in_check(self, turn=None):
        if not turn: turn = self.turn
        king_pos = self.pieces[(COLORS[turn], "k")][0].pos
        opp_color = COLORS[1-turn]
        # check if king is being attacked currently
        # pawns
        pawn_dir = -1 if turn == 0 else 1
        for update in ((pawn_dir, -1), (pawn_dir, 1)):
            new_sq = (king_pos[0] + update[0], king_pos[1] + update[1])
            if (0 <= new_sq[0] < BOARD_SIZE) and (0 <= new_sq[1] < BOARD_SIZE):
                if self.board[new_sq] is not None:
                    if self.board[new_sq].get_piece_type() == (opp_color, "p"):
                        return True
        # knight
        for update in dirs["n"]:
            new_sq = (king_pos[0] + update[0], king_pos[1] + update[1])
            if (0 <= new_sq[0] < BOARD_SIZE) and (0 <= new_sq[1] < BOARD_SIZE):
                if self.board[new_sq] is not None:
                    if self.board[new_sq].get_piece_type() == (opp_color, "n"):
                        return True
        # bishop, queen, and king
        for update in dirs["b"]:
            for i in range(1, BOARD_SIZE):
                new_sq = (king_pos[0] + i * update[0], king_pos[1] + i * update[1])
                if not ((0 <= new_sq[0] < BOARD_SIZE) and (0 <= new_sq[1] < BOARD_SIZE)):
                    break
                if self.board[new_sq] is None:
                    continue
                piece_type = self.board[new_sq].get_piece_type()
                if piece_type[0] == opp_color:
                    if piece_type[1] in ("b", "q") or (i == 1 and piece_type[1] == "k"):
                        return True
                break
        # rook, queen, and king
        for update in dirs["r"]:
            for i in range(1, BOARD_SIZE):
                new_sq = (king_pos[0] + i * update[0], king_pos[1] + i * update[1])
                if not ((0 <= new_sq[0] < BOARD_SIZE) and (0 <= new_sq[1] < BOARD_SIZE)):
                    break
                if self.board[new_sq] is None:
                    continue
                piece_type = self.board[new_sq].get_piece_type()
                if piece_type[0] == opp_color:
                    if piece_type[1] in ("r", "q") or (i == 1 and piece_type[1] == "k"):
                        return True
                break
        return False


    def _get_moves_before_illegal(self, turn, piece=None, captured_piece=None):
        if piece is not None: 
            return self._get_moves_by_piece(piece)
        moves = []
        if captured_piece is not None:  # skip this piece's moves 
            for piece_type in self._get_piece_types()[(1-turn)*6 : (2-turn)*6]:  # six piece types per color
                for piece in self.pieces[piece_type]:
                    if piece.equals(captured_piece): 
                        continue
                    moves.extend(self._get_moves_by_piece(piece))
            return moves
        for piece_type in self._get_piece_types()[(1-turn)*6 : (2-turn)*6]:  # six piece types per color
            for piece in self.pieces[piece_type]:
                moves.extend(self._get_moves_by_piece(piece))
        return moves


    def _get_moves_by_piece(self, piece, board=None):
        if not board: board = self.board
        pos = piece.pos
        moves = piece.get_moves()
        ret = []
        for row in range(moves.shape[0]):
            for move in moves[row]:
                if move == None:
                    break
                new_sq = (pos[0] + move[0], pos[1] + move[1])
                if (not (0 <= new_sq[0] < BOARD_SIZE)) or (not (0 <= new_sq[1] < BOARD_SIZE)):
                    break  # gone out of bounds along this direction
                if not isinstance(piece, Pawn):
                    if board[new_sq] != None:
                        if board[new_sq].color != piece.color:
                            ret.append((pos, new_sq))
                        break
                    ret.append((pos, new_sq))
                else:  # pawn move
                    if move[1] != 0:  # pawn capture move; needs an opposing piece there
                        if board[new_sq] != None and board[new_sq].color != piece.color:
                            ret.append((pos, new_sq))
                    elif move[1] == 0:  # pawn forward move; needs empty square
                        if board[new_sq] == None:
                            ret.append((pos, new_sq))
                    break
        return ret


    def _update_board(self, move, board = None):  # move is a tuple of the form (from_sq, to_sq)
        if not board: board = self.board  
        board[move[1]] = board[move[0]]
        board[move[0]] = None
        return board


    def _reset_history(self):
        self.history = {}
        self.history[self.board_to_string(False)] = 1


    def play_move(self, move, lookahead=False):
        curr = self.board[move[0]]
        assert curr is not None

        captured = self.board[move[1]]
        self.board = self._update_board(move)  # update board
        curr.play_move(move)   # update moved piece
        if lookahead:  
            return captured  # doesn't change self.pieces and returns piece to ignore for lookahead
        
        reset_history = False
        if captured is not None:  # remove captured piece
            self.pieces[captured.get_piece_type()].remove(captured)
            reset_history = True

        if isinstance(curr, Pawn):
            reset_history = True
            if ((move[1][0] == BOARD_SIZE-1 and curr.color == "b") or (move[1][0] == 0 and curr.color == "w")): # pawn promotion move
                queen = Queen(curr.color, curr.pos)
                self.board[queen.pos] = queen
                self.pieces[(curr.color, "q")].append(queen)  # add queen to self.pieces
                self.pieces[curr.get_piece_type()].remove(curr)  # remove pawn

        # and the housekeeping
        self.turn = 1-self.turn
        self.move_number += 1
        # when playing move, only need to check that current state generates threefold draw.
        if reset_history:          
            self._reset_history()  # captures and pawn moves cannot be undone, so no previous history can be duplicated
        else:
            state = self.board_to_string(False)
            self.history[state] = self.history.get(state, 0) + 1
            if self.history[state] >= 3:
                return "Draw by threefold repitition."
                
        term = self.is_terminal(False)
        if term: 
            return term


    def is_terminal(self, check_threefold=True):
        if not self._has_moves():
            if self._in_check():
                return str("White" if self.turn==1 else "Black") + " wins by checkmate."
            return "Draw by stalemate."

        elif self.move_number > 160:
            return "Draw by 80-move rule."

        insufficient = True
        for color in COLORS:
            for piece in ("q", "r", "p"):
                if len(self.pieces[(color, piece)]) > 0:
                    insufficient = False
                    break
            if insufficient == False: 
                break
            if len(self.pieces[(color, "b")]) + len(self.pieces[(color, "n")]) > 1:
                insufficient = False
                break
        if insufficient:
            return "Draw by insufficient material."

        if check_threefold:
            if any(count >= 3 for count in self.history.values()):
                return "Draw by threefold repitition."


    def board_to_string(self, move_num=True):
        state = ""
        if move_num: 
            state += str(self.move_number) + '!'
        state += COLORS[self.turn]
        for piece_type in self._get_piece_types():
            state += piece_type[0] + piece_type[1]
            ps = []  # get all positions of each piece type, sort, and add to string
            for piece in self.pieces[piece_type]:
                ps.append(piece.pos)
            ps.sort()
            for p in ps:
                state += str(p)
        return state


    def hist_to_string(self):
        state = ""
        for s in self.history.keys():
            state += "!" + str(self.history[s]) + s
        return state


    def state_to_string(self):
        return self.board_to_string() + self.hist_to_string()


    def from_string(self, state):
        assert BOARD_SIZE <= 10  # after 10, positions can be double digit numbers and string parsing logic needs to be updated

        i = state.index('!')
        move_number = int(state[:i])
        assert move_number < 161  # should be less than 160 by 80 move rule
        i += 1
        turn = COLORS.index(state[i:i+1])
        i += 1
        
        pieces = {}
        for piece_type in PIECE_TYPES:
            pieces[piece_type] = []
        board = np.full((BOARD_SIZE, BOARD_SIZE), fill_value=None)

        while not state[i:i+1] == '!':
            piece_type = (state[i:i+1], state[i+1:i+2])
            assert piece_type in PIECE_TYPES  # ensure its a valid piece
            i += 2
            while state[i:i+1] == '(':
                pos = (int(state[i+1:i+2]), int(state[i+4:i+5]))
                if piece_type[1] == "k":
                    piece = King(piece_type[0], pos)
                elif piece_type[1] == "q":
                    piece = Queen(piece_type[0], pos)
                elif piece_type[1] == "b":
                    piece = Bishop(piece_type[0], pos)
                elif piece_type[1] == "n":
                    piece = Knight(piece_type[0], pos)
                elif piece_type[1] == "r":
                    piece = Rook(piece_type[0], pos)
                elif piece_type[1] == "p":
                    piece = Pawn(piece_type[0], pos)
                pieces[piece_type].append(piece)
                # board[pos] = piece
                i += 6

        board = self._create_board([piece for piece_type in pieces.values() for piece in piece_type])

        self.move_number = move_number
        self.turn = turn
        self.pieces = pieces
        self.board = board
        
        # once reaches another !, delineates start of history
        assert state[i:i+1] == '!'
        state = state[i+1:]

        if len(state) == 0:  # if passed in only board state
            self._reset_history()
        else:  # if history is stored
            history = {}
            while '!' in state:
                i = state.index("!")
                history[state[1:i]] = int(state[:1])
                state = state[i+1:]  # skip "!"
            if len(state) > 0:  # should have 1 last game state
                history[state[1:]] = int(state[:1])

            self.history = history

        
                
                





                           

    
