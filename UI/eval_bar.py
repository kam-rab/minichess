"""The eval bar: a vertical bar beside the board showing how good the position is for White.

The evaluation comes from an MCTS agent searching the current position on a background thread. The value
shown is the Q of the search's most visited move (the move it would play), as Leela shows it. Unlike the
average over all the root's visits, it isn't pulled down by visits spent ruling out worse moves. Q is in
[-1, 1] from the side to move's view; the bar shows it from White's view (+1 = White winning, -1 = Black
winning). A proven mate overrides it with exactly +-1.
"""

import numpy as np
import pygame

WHITE_SIDE = (236, 236, 236)
BLACK_SIDE = (44, 44, 48)
EDGE = (100, 105, 118)
MIDLINE = (150, 60, 60)
MAX_PLIES = 160


def summarize(root, sims, has_moves_left):
    """
    Read the search tree. `sims` is the number of new simulations this search has run (not shown: the
    tree may also hold visits reused from earlier searches). Returns a dict with
        q: Q of the most visited move from White's view, in [-1, 1] (+-1 if a mate is proven)
        q_avg: average Q over all the root's visits, from White's view (not shown; kept for debugging)
        sims: the tree's total visits at the root (reused + new): the size of the search being shown
        depth: length of the principal variation (most-visited line) in plies
        plies_left: predicted plies until the game ends (agents with a moves-left head), else None
        mate: forced mate the search has proven, as a signed number of moves from White's view
            (+3 = White mates in 3, -2 = Black mates in 2), else None
    """
    visits = root.children_visits.sum()
    q_avg = root.children_values.sum() / visits if visits else 0.0      # side to move's view, all visits
    q_root = q_avg
    if visits:      # the eval is the Q of the most visited move (what the search would play), as in Leela
        best = int(np.argmax(root.children_visits))
        q_root = root.children_values[best] / root.children_visits[best]
    depth, node = 0, root
    while node is not None and node.terminal is None and len(node.children_visits) and node.children_visits.sum() > 0:
        node = node.children[int(np.argmax(node.children_visits))]
        depth += 1
    plies_left = None
    if has_moves_left and visits:
        plies_left = root.children_moves_left.sum() / visits * MAX_PLIES
    mate = None
    proven, proven_plies = getattr(root, "proven", None), getattr(root, "proven_plies", None)
    if proven is not None and proven_plies is not None:
        mate = (proven_plies + 1) // 2 * proven * root.color   # winner's moves until mate, signed for White
        q_root = proven                                         # exact result: full bar
        plies_left = float(proven_plies)
    return {"q": float(q_root * root.color), "q_avg": float(q_avg * root.color), "sims": int(visits), "depth": depth,
            "plies_left": plies_left, "mate": mate}


def _mate_through(node):
    """Signed mate (White's view) after the move into child `node`, counted from before that move; None if unproven."""
    if node is None or getattr(node, "proven", None) is None or node.proven_plies is None:
        return None
    plies = node.proven_plies + 1                 # one more ply: the move into node
    winner = node.proven * node.color             # node.proven is from node's side to move: +1 means node.color wins
    return (plies + 1) // 2 * winner


def _next_on_line(node):
    """The child index the principal variation follows from node: the proven-best move if node is proven,
    otherwise the most visited move. None if the line ends here."""
    if node is None or node.terminal is not None or not len(node.children_visits) or node.children_visits.sum() == 0:
        return None
    proven = getattr(node, "proven", None)
    if proven is not None:
        if proven == 1:      # winning: the fastest mate
            options = [i for i, c in enumerate(node.children) if c is not None and c.proven == -1]
            return min(options, key=lambda i: node.children[i].proven_plies)
        if all(c is not None and c.proven_plies is not None for c in node.children):    # lost: the longest defence
            return max(range(len(node.children)), key=lambda i: node.children[i].proven_plies)
    return int(np.argmax(node.children_visits))


def engine_lines(root, state, n_lines=3, max_plies=8, min_visits=2):
    """
    The search's top n_lines root moves, each with its continuation, for display.
    state: a State for the root position (not modified). Returns a list of dicts, best first:
        q: that move's value from White's view
        mate: signed proven mate through that move (White's view), else None
        text: the line in numbered notation, e.g. "16. Qc4+ Ka3 17. Ra1 Kb2 18. Qc1#"
    A line stops after max_plies, at the end of the game, or where the search spent fewer than min_visits.
    """
    import copy
    from Model.util import decode
    from UI.notation import play_labeled

    visits = root.children_visits
    order = [i for i in np.argsort(-visits, kind="stable") if visits[i] > 0][:n_lines]
    lines = []
    for i in order:
        q = root.children_values[i] / visits[i] * root.color
        mate = _mate_through(root.children[i])
        board = copy.deepcopy(state)
        parts, node, a = [], root, i
        for ply in range(max_plies):
            turn = (1 - node.color) // 2
            if turn == 0 or ply == 0:
                number = board.move_number // 2 + 1
                parts.append(f"{number}." if turn == 0 else f"{number}...")
            label, result = play_labeled(board, decode(node.moves[a], turn))
            parts.append(label)
            node = node.children[a]
            if result is not None or node is None:
                break
            a = _next_on_line(node)
            if a is None or (node.proven is None and node.children_visits[a] < min_visits):
                break
        lines.append({"q": float(q), "mate": mate, "text": " ".join(parts)})
    return lines


def draw_eval_bar(surf, rect, q, flipped, font, label=None):
    """
    Draw the bar in rect. q is White's evaluation in [-1, 1], or None (nothing to show yet).
    White's share fills from White's side of the board: the bottom normally, the top when the board is flipped.
    label: text to show instead of the number (e.g. "M3" for a proven mate).
    """
    rect = pygame.Rect(rect)
    pygame.draw.rect(surf, BLACK_SIDE, rect)
    if q is not None:
        white_h = int(round(rect.h * (1 + max(-1.0, min(1.0, q))) / 2))
        white = pygame.Rect(rect.x, rect.top if flipped else rect.bottom - white_h, rect.w, white_h)
        pygame.draw.rect(surf, WHITE_SIDE, white)
        # the number, written at the end of the bar belonging to the side that's ahead
        if label is None:
            label = f"{abs(q):.2f}".lstrip("0") if abs(q) < 1 else "1.0"
        text = font.render(label, True, BLACK_SIDE if q >= 0 else WHITE_SIDE)
        white_end_bottom = not flipped
        at_bottom = white_end_bottom if q >= 0 else not white_end_bottom
        pos = text.get_rect(midbottom=(rect.centerx, rect.bottom - 4)) if at_bottom else \
            text.get_rect(midtop=(rect.centerx, rect.top + 4))
        surf.blit(text, pos)
    pygame.draw.line(surf, MIDLINE, (rect.left, rect.centery), (rect.right - 1, rect.centery))
    pygame.draw.rect(surf, EDGE, rect, width=1)
