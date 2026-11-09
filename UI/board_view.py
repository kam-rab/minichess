"""Draws the board and pieces, and converts between screen pixels and engine squares.

Engine squares are (row, col) with (0, 0) at the top left from White's point of view.
Row 0 is rank 5 and col 0 is file a.
"""

import pygame

from Engine.pieces import BOARD_SIZE, Bishop, King, Knight, Pawn, Queen, Rook

LIGHT = (238, 221, 190)
DARK = (176, 134, 98)
SELECTED = (246, 222, 92)
LAST_MOVE = (205, 210, 106)
CHECK = (220, 60, 60)
HINT = (40, 40, 40, 70)  # translucent
COORD_TEXT = (200, 200, 205)
WHITE_FILL, WHITE_EDGE = (250, 250, 250), (25, 25, 25)
BLACK_FILL = (25, 25, 25)

FILES = "abcde"
# (outline glyph, solid glyph) per piece type
GLYPHS = {
    King: ("♔", "♚"), Queen: ("♕", "♛"), Rook: ("♖", "♜"),
    Bishop: ("♗", "♝"), Knight: ("♘", "♞"), Pawn: ("♙", "♟"),
}
LETTERS = {King: "K", Queen: "Q", Rook: "R", Bishop: "B", Knight: "N", Pawn: "P"}
GLYPH_FONTS = ["applesymbols", "dejavusans", "segoeuisymbol", "notosanssymbols2", "freeserif"]


def square_name(sq):
    return f"{FILES[sq[1]]}{BOARD_SIZE - sq[0]}"


def _font_has_glyph(font, char):
    """pygame substitutes a placeholder box for missing glyphs, so compare against a surely-missing char."""
    missing = pygame.image.tobytes(font.render("", True, (0, 0, 0)), "RGBA")
    return pygame.image.tobytes(font.render(char, True, (0, 0, 0)), "RGBA") != missing


def _load_glyph_font(size):
    for name in GLYPH_FONTS:
        path = pygame.font.match_font(name)
        if path:
            font = pygame.font.Font(path, size)
            if all(_font_has_glyph(font, g) for pair in GLYPHS.values() for g in pair):
                return font
    return None


class BoardView:
    def __init__(self, topleft, square_size):
        self.x, self.y = topleft
        self.sq = square_size
        self.flipped = False  # True shows Black at the bottom
        self.rect = pygame.Rect(self.x, self.y, square_size * BOARD_SIZE, square_size * BOARD_SIZE)
        self.glyph_font = _load_glyph_font(int(square_size * 0.82))
        self.letter_font = pygame.font.SysFont("helvetica,arial", int(square_size * 0.42), bold=True)
        self.coord_font = pygame.font.SysFont("helvetica,arial", 15)
        self.hint = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
        self._piece_cache = {}

    # coordinates

    def square_rect(self, sq):
        row, col = sq
        if self.flipped:
            row, col = BOARD_SIZE - 1 - row, BOARD_SIZE - 1 - col
        return pygame.Rect(self.x + col * self.sq, self.y + row * self.sq, self.sq, self.sq)

    def square_at(self, pos):
        if not self.rect.collidepoint(pos):
            return None
        col = (pos[0] - self.x) // self.sq
        row = (pos[1] - self.y) // self.sq
        if self.flipped:
            row, col = BOARD_SIZE - 1 - row, BOARD_SIZE - 1 - col
        return (row, col)

    # drawing

    def draw(self, surf, board, selected=None, targets=(), last_move=None, check_sq=None):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                sq = (row, col)
                rect = self.square_rect(sq)
                color = DARK if (row + col) % 2 == 0 else LIGHT
                if last_move and sq in last_move:
                    color = _blend(color, LAST_MOVE, 0.55)
                if sq == selected:
                    color = _blend(color, SELECTED, 0.7)
                pygame.draw.rect(surf, color, rect)
                if sq == check_sq:
                    pygame.draw.circle(surf, CHECK, rect.center, self.sq // 2 - 4)
                piece = board[sq]
                if piece is not None:
                    self._draw_piece(surf, piece, rect)
                if sq in targets:
                    self._draw_target(surf, rect, capture=piece is not None)
        self._draw_coords(surf)

    def _draw_target(self, surf, rect, capture):
        self.hint.fill((0, 0, 0, 0))
        center = (self.sq // 2, self.sq // 2)
        if capture:
            pygame.draw.circle(self.hint, HINT, center, self.sq // 2 - 2, width=max(4, self.sq // 14))
        else:
            pygame.draw.circle(self.hint, HINT, center, self.sq // 7)
        surf.blit(self.hint, rect.topleft)

    def _draw_piece(self, surf, piece, rect):
        key = (type(piece), piece.color)
        if key not in self._piece_cache:
            self._piece_cache[key] = self._render_piece(*key)
        image = self._piece_cache[key]
        surf.blit(image, image.get_rect(center=rect.center))

    def _render_piece(self, kind, color):
        if self.glyph_font is not None:
            outline, solid = GLYPHS[kind]
            if color == "w":  # white fill under a dark outline
                base = self.glyph_font.render(solid, True, WHITE_FILL)
                base.blit(self.glyph_font.render(outline, True, WHITE_EDGE), (0, 0))
                return base
            return self.glyph_font.render(solid, True, BLACK_FILL)
        # fallback when no font has chess glyphs: a disc with a letter
        size = int(self.sq * 0.78)
        image = pygame.Surface((size, size), pygame.SRCALPHA)
        fill, ink = (WHITE_FILL, WHITE_EDGE) if color == "w" else (BLACK_FILL, WHITE_FILL)
        pygame.draw.circle(image, fill, (size // 2, size // 2), size // 2)
        pygame.draw.circle(image, WHITE_EDGE, (size // 2, size // 2), size // 2, width=2)
        letter = self.letter_font.render(LETTERS[kind], True, ink)
        image.blit(letter, letter.get_rect(center=(size // 2, size // 2)))
        return image

    def _draw_coords(self, surf):
        for i in range(BOARD_SIZE):
            col_rect = self.square_rect((BOARD_SIZE - 1, i) if not self.flipped else (0, i))
            text = self.coord_font.render(FILES[i], True, COORD_TEXT)
            surf.blit(text, text.get_rect(midtop=(col_rect.centerx, self.rect.bottom + 6)))
            row_rect = self.square_rect((i, 0) if not self.flipped else (i, BOARD_SIZE - 1))
            text = self.coord_font.render(str(BOARD_SIZE - i), True, COORD_TEXT)
            surf.blit(text, text.get_rect(midright=(self.rect.left - 8, row_rect.centery)))


def _blend(a, b, t):
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))
