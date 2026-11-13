"""Move notation shared by the move list and the eval's engine lines."""

from Engine.pieces import Bishop, King, Knight, Pawn, Queen, Rook
from UI.board_view import square_name

LETTERS = {King: "K", Queen: "Q", Rook: "R", Bishop: "B", Knight: "N", Pawn: ""}


def algebraic(board, move, legal):
    """Standard algebraic notation for a move about to be played, e.g. "Nc3", "exd4", "Rab1", "Q1e2".

    The starting square is added only when another piece of the same type could also reach the target:
    its file if that tells them apart, otherwise its rank, otherwise both. Pawn captures always give the
    file the pawn came from, as in real chess. Promotion and check suffixes are added by the caller.
    """
    (from_sq, to_sq), piece = move, board[move[0]]
    capture = "x" if board[to_sq] is not None else ""
    if isinstance(piece, Pawn):
        return (square_name(from_sq)[0] + capture if capture else "") + square_name(to_sq)

    rivals = [m[0] for m in legal
              if m[1] == to_sq and m[0] != from_sq and type(board[m[0]]) is type(piece)]
    origin = ""
    if rivals:
        if all(sq[1] != from_sq[1] for sq in rivals):
            origin = square_name(from_sq)[0]  # file
        elif all(sq[0] != from_sq[0] for sq in rivals):
            origin = square_name(from_sq)[1:]  # rank
        else:
            origin = square_name(from_sq)
    return f"{LETTERS[type(piece)]}{origin}{capture}{square_name(to_sq)}"


def play_labeled(state, move):
    """Play move on state (modifying it) and return (label, result): its full notation including
    promotion (=Q), check (+) and mate (#), and play_move's result (None unless the game ended)."""
    legal = state.get_moves()
    label = algebraic(state.board, move, legal)
    piece = state.board[move[0]]
    result = state.play_move(move)
    if isinstance(piece, Pawn) and isinstance(state.board[move[1]], Queen):
        label += "=Q"
    if result and "checkmate" in str(result):
        label += "#"
    elif state._in_check():
        label += "+"
    return label, result
