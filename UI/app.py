"""Main minichess window: board on the left, controls and move list on the right.

Each side is a human (who clicks on the board) or an agent from Agents/. Agents choose their moves on a
background thread using a copy of the game state, so the window keeps responding while they think.
"""

import copy
import queue
import threading
import time
import traceback

import pygame

from Engine.pieces import BOARD_SIZE, Bishop, King, Knight, Pawn, Queen, Rook
from Engine.state import STATE_INIT, State
from UI.board_view import BoardView, square_name
from UI.players import HUMAN, discover_agents
from UI.widgets import TEXT, TEXT_DIM, Button, Selector

COLORS = ("w", "b")
COLOR_NAMES = ("White", "Black")
LETTERS = {King: "K", Queen: "Q", Rook: "R", Bishop: "B", Knight: "N", Pawn: ""}

SQUARE = 96
MARGIN = 34
PANEL_W = 300
BOARD_PX = SQUARE * BOARD_SIZE
WIDTH = MARGIN + BOARD_PX + MARGIN + PANEL_W
HEIGHT = max(MARGIN + BOARD_PX + MARGIN, 580)
AGENT_DELAY = 0.4  # minimum seconds between an agent's move and the next one, so games can be watched

BG = (38, 40, 46)
ERROR = (240, 110, 100)
ACCENT = (246, 222, 92)


class App:
    def __init__(self, start_state=STATE_INIT, white=HUMAN, black=HUMAN):
        pygame.init()
        pygame.display.set_caption("Minichess")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("helvetica,arial", 17)
        self.small = pygame.font.SysFont("helvetica,arial", 14)
        self.title_font = pygame.font.SysFont("helvetica,arial", 26, bold=True)
        self.mono = pygame.font.SysFont("menlo,consolas,dejavusansmono,courier", 15)

        self.start_state = start_state
        self.view = BoardView((MARGIN, MARGIN), SQUARE)
        self.agent_factories, self.agent_errors = discover_agents()
        self.agents = [None, None]  # agent instance per color, or None for a human
        self.results = queue.Queue()  # (token, move, error) from agent threads
        self.token = 0  # increments whenever pending agent results become stale
        self.thinking = False
        self.paused = False
        self.message = None  # last error to show the user
        self.running = True

        self._build_panel()
        self.new_game()
        for color, name in ((0, white), (1, black)):
            if name != HUMAN:
                if name not in self.selectors[color].options:
                    self.message = f"No agent named '{name}'."
                    continue
                self.selectors[color].set_value(name)
                self.set_player(color, name)

    # ---------------------------------------------------------------- game control

    def new_game(self):
        self.game = State(self.start_state)
        self.snapshots = []  # (state, last_move, result) before each move, for undo
        self.move_labels = []
        self.last_move = None
        self.message = None
        self.result = self._initial_result()
        self.next_agent_time = 0.0
        self._invalidate_agent()
        self._clear_selection()
        # fresh agent instances so agents can keep per-game state
        for color in (0, 1):
            if self.agents[color] is not None:
                self.set_player(color, self.selectors[color].value)

    def _initial_result(self):
        try:
            # Repetition is skipped here: a freshly loaded State stores its history counts as strings.
            return self.game.is_terminal(check_threefold=False)
        except Exception as e:
            self.message = f"is_terminal failed: {type(e).__name__}: {e}"
            return None

    def set_player(self, color, name):
        self._invalidate_agent()
        self._clear_selection()
        if name == HUMAN:
            self.agents[color] = None
        else:
            try:
                self.agents[color] = self.agent_factories[name]()
            except Exception as e:
                traceback.print_exc()
                self.message = f"Could not create {name}: {type(e).__name__}: {e}"
                self.agents[color] = None
                self.selectors[color].set_value(HUMAN)
        # by default, put a lone human player at the bottom of the board
        self.view.flipped = self.agents[0] is not None and self.agents[1] is None

    def undo(self):
        if not self.snapshots:
            return
        self._invalidate_agent()
        self._restore(self.snapshots.pop())
        # against an agent, step back to the human's turn rather than letting the agent replay at once
        if self._is_agent_turn() and self._has_human() and self.snapshots:
            self._restore(self.snapshots.pop())
        if not self._has_human():
            self.paused = True

    def _restore(self, snapshot):
        self.game, self.last_move, self.result = snapshot
        self.move_labels.pop()
        self.message = None
        self._clear_selection()

    def toggle_pause(self):
        self.paused = not self.paused
        self.next_agent_time = 0.0

    def flip(self):
        self.view.flipped = not self.view.flipped

    def play(self, move):
        """Play a move on the real game, keeping an undo snapshot. Returns False if it was rejected."""
        legal = self.game.get_moves()
        if move not in legal:
            self.message = f"Illegal move {_move_text(move)} for {COLOR_NAMES[self.game.turn]}."
            return False
        label = _algebraic(self.game.board, move, legal)
        piece = self.game.board[move[0]]
        before = (copy.deepcopy(self.game), self.last_move, self.result)
        try:
            result = self.game.play_move(move)
        except Exception as e:
            traceback.print_exc()
            self.game = before[0]  # play_move may have half-applied the move
            self.message = f"play_move failed: {type(e).__name__}: {e}"
            self.paused = True
            return False

        if isinstance(piece, Pawn) and isinstance(self.game.board[move[1]], Queen):
            label += "=Q"
        if result and "checkmate" in str(result):
            label += "#"
        elif self._check_square():
            label += "+"
        self.snapshots.append(before)
        self.move_labels.append(label)
        self.last_move = move
        self.result = result
        self.message = None
        self.next_agent_time = time.monotonic() + AGENT_DELAY
        self._clear_selection()
        return True

    # ---------------------------------------------------------------- humans

    def click_board(self, pos):
        sq = self.view.square_at(pos)
        if sq is None or self.result or self._is_agent_turn():
            return
        if sq in self.targets:
            self.play(self.targets[sq])
            return
        piece = self.game.board[sq]
        if piece is not None and piece.color == COLORS[self.game.turn] and sq != self.selected:
            self.selected = sq
            self.targets = {move[1]: move for move in self.game.get_moves(piece)}
        else:
            self._clear_selection()

    def _clear_selection(self):
        self.selected = None
        self.targets = {}

    # ---------------------------------------------------------------- agents

    def _is_agent_turn(self):
        return self.agents[self.game.turn] is not None

    def _has_human(self):
        return self.agents[0] is None or self.agents[1] is None

    def _invalidate_agent(self):
        self.token += 1
        self.thinking = False

    def _maybe_start_agent(self):
        if self.result or self.paused or self.thinking or not self._is_agent_turn():
            return
        if time.monotonic() < self.next_agent_time:
            return
        self.thinking = True
        agent, token = self.agents[self.game.turn], self.token
        state = copy.deepcopy(self.game)  # the agent may modify its copy as it searches

        def work():
            try:
                self.results.put((token, agent.get_move(state), None))
            except Exception as e:
                traceback.print_exc()
                self.results.put((token, None, f"{type(e).__name__}: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _collect_agent_moves(self):
        while not self.results.empty():
            token, move, error = self.results.get()
            if token != self.token:
                continue  # game changed while the agent was thinking
            self.thinking = False
            name = self.selectors[self.game.turn].value
            if error:
                self.message = f"{name} crashed: {error}"
                self.paused = True
                continue
            try:
                move = _normalize(move)
            except Exception:
                self.message = f"{name} returned {move!r}, expected ((row, col), (row, col))."
                self.paused = True
                continue
            if not self.play(move):
                self.message = f"{name}: {self.message}"
                self.paused = True

    # ---------------------------------------------------------------- panel

    def _build_panel(self):
        x = MARGIN + BOARD_PX + MARGIN
        w = PANEL_W - MARGIN
        half = (w - 10) // 2
        options = [HUMAN] + list(self.agent_factories)
        self.panel_x, self.panel_w = x, w
        self.selectors = [
            Selector((x + 62, 84 + 46 * color, w - 62, 34), options,
                     lambda name, color=color: self.set_player(color, name))
            for color in (0, 1)
        ]
        self.buttons = [
            Button((x, 186, half, 36), "New game", self.new_game),
            Button((x + half + 10, 186, half, 36), "Undo", self.undo, lambda: bool(self.snapshots)),
            Button((x, 230, half, 36), lambda: "Resume" if self.paused else "Pause", self.toggle_pause,
                   lambda: self.paused or any(agent is not None for agent in self.agents)),
            Button((x + half + 10, 230, half, 36), "Flip board", self.flip),
        ]

    def _status_lines(self):
        turn = self.game.turn
        side = f"{COLOR_NAMES[turn]} ({self.selectors[turn].value})"
        if self.result:
            return [("Game over", ACCENT), (str(self.result), TEXT)]
        if self.thinking:
            return [(f"{side} is thinking...", TEXT)]
        if self.paused and self._is_agent_turn():
            return [(f"{side} to move", TEXT), ("Paused - press Space to resume", TEXT_DIM)]
        check = " - in check" if self._check_square() else ""
        return [(f"{side} to move{check}", TEXT)]

    def _check_square(self):
        try:
            if self.game._in_check():
                return self.game.pieces[(COLORS[self.game.turn], "k")][0].pos
        except Exception:
            pass
        return None

    # ---------------------------------------------------------------- drawing

    def draw(self):
        self.screen.fill(BG)
        mouse = pygame.mouse.get_pos()
        self.view.draw(self.screen, self.game.board, self.selected, self.targets, self.last_move,
                       self._check_square())

        x, w = self.panel_x, self.panel_w
        self.screen.blit(self.title_font.render("Minichess", True, TEXT), (x, MARGIN - 4))
        for color, selector in enumerate(self.selectors):
            label = self.font.render(COLOR_NAMES[color], True, TEXT)
            self.screen.blit(label, label.get_rect(midleft=(x, selector.rect.centery)))
            selector.draw(self.screen, self.font, mouse)
        for button in self.buttons:
            button.draw(self.screen, self.font, mouse)

        y = 284
        for text, color in self._status_lines():
            for line in _wrap(text, self.font, w):
                self.screen.blit(self.font.render(line, True, color), (x, y))
                y += 21
        if self.message:
            for line in _wrap(self.message, self.small, w):
                self.screen.blit(self.small.render(line, True, ERROR), (x, y))
                y += 18
        self._draw_moves(x, max(y + 14, 352), w)
        self._draw_footer(x, w)
        pygame.display.flip()

    def _draw_moves(self, x, y, w):
        self.screen.blit(self.font.render("Moves", True, TEXT_DIM), (x, y))
        y += 26
        start_number = int(self.start_state.split("!", 1)[0]) // 2 + 1
        rows = []
        labels = self.move_labels
        offset = 1 if self.start_state.split("!", 2)[1][0] == "b" else 0  # game started with Black to move
        padded = [".."] * offset + labels
        for i in range(0, len(padded), 2):
            rows.append(f"{start_number + i // 2:>3}. {padded[i]:<10}{padded[i + 1] if i + 1 < len(padded) else ''}")
        bottom = HEIGHT - 52
        visible = max(1, (bottom - y) // 19)
        for row in rows[-visible:]:
            self.screen.blit(self.mono.render(row, True, TEXT), (x, y))
            y += 19

    def _draw_footer(self, x, w):
        found = len(self.agent_factories)
        text = f"{found} agent{'s' if found != 1 else ''} found in Agents/"
        if self.agent_errors:
            text += f" - failed to import: {', '.join(self.agent_errors)} (see terminal)"
        lines = _wrap(text, self.small, w)
        y = HEIGHT - 14 - 18 * len(lines)
        for line in lines:
            self.screen.blit(self.small.render(line, True, ERROR if self.agent_errors else TEXT_DIM), (x, y))
            y += 18

    # ---------------------------------------------------------------- main loop

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for widget in self.selectors + self.buttons:
                if widget.handle_click(event.pos):
                    return
            self.click_board(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                self.new_game()
            elif event.key in (pygame.K_u, pygame.K_BACKSPACE):
                self.undo()
            elif event.key == pygame.K_f:
                self.flip()
            elif event.key == pygame.K_SPACE:
                self.toggle_pause()
            elif event.key == pygame.K_ESCAPE:
                self._clear_selection()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            self._collect_agent_moves()
            self._maybe_start_agent()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


def _normalize(move):
    """Accept lists, numpy ints, etc. and return ((row, col), (row, col)) of plain ints."""
    (r0, c0), (r1, c1) = move
    return ((int(r0), int(c0)), (int(r1), int(c1)))


def _algebraic(board, move, legal):
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


def _move_text(move):
    try:
        return f"{square_name(move[0])}-{square_name(move[1])}"
    except Exception:
        return repr(move)


def _wrap(text, font, width):
    lines, line = [], ""
    for word in str(text).split():
        candidate = f"{line} {word}".strip()
        if line and font.size(candidate)[0] > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line] if line else lines
