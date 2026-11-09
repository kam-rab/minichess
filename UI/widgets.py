"""Minimal pygame widgets for the side panel: buttons and a left/right option selector."""

import pygame

TEXT = (232, 232, 236)
TEXT_DIM = (130, 133, 142)
BUTTON = (68, 72, 82)
BUTTON_HOVER = (88, 93, 106)
BUTTON_DISABLED = (52, 54, 61)
BORDER = (100, 105, 118)


def _resolve(value):
    return value() if callable(value) else value


class Button:
    def __init__(self, rect, label, on_click, enabled=True):
        self.rect = pygame.Rect(rect)
        self.label = label  # str, or a function returning str
        self.on_click = on_click
        self.enabled = enabled  # bool, or a function returning bool

    def draw(self, surf, font, mouse_pos):
        enabled = _resolve(self.enabled)
        if not enabled:
            color = BUTTON_DISABLED
        elif self.rect.collidepoint(mouse_pos):
            color = BUTTON_HOVER
        else:
            color = BUTTON
        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        text = font.render(_resolve(self.label), True, TEXT if enabled else TEXT_DIM)
        surf.blit(text, text.get_rect(center=self.rect.center))

    def handle_click(self, pos):
        if self.rect.collidepoint(pos) and _resolve(self.enabled):
            self.on_click()
            return True
        return False


class Selector:
    """Shows one option from a list; clicking the left/right arrows cycles through them."""

    ARROW_W = 30

    def __init__(self, rect, options, on_change, index=0):
        self.rect = pygame.Rect(rect)
        self.options = list(options)
        self.index = index
        self.on_change = on_change  # called with the newly selected option
        self.left = pygame.Rect(self.rect.x, self.rect.y, self.ARROW_W, self.rect.h)
        self.right = pygame.Rect(self.rect.right - self.ARROW_W, self.rect.y, self.ARROW_W, self.rect.h)

    @property
    def value(self):
        return self.options[self.index]

    def set_value(self, value):
        self.index = self.options.index(value)

    def draw(self, surf, font, mouse_pos):
        pygame.draw.rect(surf, BUTTON, self.rect, border_radius=6)
        many = len(self.options) > 1
        for arrow, glyph in ((self.left, "<"), (self.right, ">")):
            if many and arrow.collidepoint(mouse_pos):
                pygame.draw.rect(surf, BUTTON_HOVER, arrow, border_radius=6)
            text = font.render(glyph, True, TEXT if many else TEXT_DIM)
            surf.blit(text, text.get_rect(center=arrow.center))
        label = font.render(self.value, True, TEXT)
        surf.blit(label, label.get_rect(center=self.rect.center))

    def handle_click(self, pos):
        if len(self.options) < 2:
            return False
        if self.left.collidepoint(pos):
            step = -1
        elif self.right.collidepoint(pos):
            step = 1
        else:
            return False
        self.index = (self.index + step) % len(self.options)
        self.on_change(self.value)
        return True
