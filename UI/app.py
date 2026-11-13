"""Main minichess window: board on the left, controls and move list on the right.

Each side is a human (who clicks on the board) or an agent from Agents/. Agents choose their moves on a
background thread using a copy of the game state, so the window keeps responding while they think.

The eval bar on the left shows an MCTS agent's evaluation of the current position (picked with the Eval selector).
It searches on its own background thread and updates as the search grows, until the position changes.

The Book selector starts new games from a book opening (Eval/openings.json, or the file given with --book, e.g. an
eval's saved openings), to see which openings the evals use.
"""

import copy
import queue
import threading
import time
import traceback

import pygame

from Engine.pieces import BOARD_SIZE, Pawn, Queen
from Engine.state import STATE_INIT, State
from Eval.opening_book import BOOK_PATH, load_book
from Model.util import terminal_value
from UI.board_view import BoardView, square_name
from UI.notation import algebraic as _algebraic
from UI.eval_bar import draw_eval_bar, engine_lines, summarize
from UI.players import HUMAN, discover_agents
from UI.widgets import TEXT, TEXT_DIM, Button, Selector

COLORS = ("w", "b")
COLOR_NAMES = ("White", "Black")

SQUARE = 96
MARGIN = 34
PANEL_W = 300
BOARD_PX = SQUARE * BOARD_SIZE
EVAL_X, EVAL_W = 16, 28   # eval bar, left of the board (the rank labels sit between the two)
BOARD_X = EVAL_X + EVAL_W + 30
WIDTH = BOARD_X + BOARD_PX + MARGIN + PANEL_W
HEIGHT = max(MARGIN + BOARD_PX + MARGIN, 680)   # tall enough for two Sims rows and the move list
# engine lines: shown under the board while the eval is on (the window grows to fit them)
ENGINE_LINES = 3              # how many of the search's top moves to show
ENGINE_LINE_PLIES = 8         # longest continuation shown
ENGINE_LINES_EVERY = 0.3      # seconds between rebuilding the lines during a search (they cost a little time)
LINES_Y = MARGIN + BOARD_PX + 36
LINE_H = 24
HEIGHT_EVAL = max(HEIGHT, LINES_Y + ENGINE_LINES * LINE_H + 24)
AGENT_DELAY = 0.4  # minimum seconds between an agent's move and the next one, so games can be watched

EVAL_OFF = "Off"
BOOK_OFF = "Off"          # Book selector: start from the normal position instead of a book opening
EVAL_MAX_SIMS = 20000     # the eval search stops here (or when the position changes)

# simulations per move (total tree visits) for MCTS agents playing in the UI, set per side with its Sims selector
MOVE_SIMS_OPTIONS = [100, 250, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000]
MOVE_SIMS_DEFAULT = 3000

# panel layout: selector rows from the top; a side's Sims row sits under it when that side is an MCTS agent
ROWS_TOP = 84
ROW_H = 46          # player / eval rows
SIMS_ROW_H = 30     # height of the smaller Sims selectors
SIMS_GAP_ABOVE = 4  # a Sims row hugs the player row it belongs to...
SIMS_GAP_BELOW = 16 # ...and leaves more room before the next row

BG = (38, 40, 46)
ERROR = (240, 110, 100)
ACCENT = (246, 222, 92)


class App:
    def __init__(self, start_state=STATE_INIT, white=HUMAN, black=HUMAN, book=BOOK_PATH):
        pygame.init()
        pygame.display.set_caption("Minichess")
        self.height = HEIGHT
        self.screen = pygame.display.set_mode((WIDTH, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("helvetica,arial", 17)
        self.small = pygame.font.SysFont("helvetica,arial", 14)
        self.title_font = pygame.font.SysFont("helvetica,arial", 26, bold=True)
        self.mono = pygame.font.SysFont("menlo,consolas,dejavusansmono,courier", 15)

        self.start_state = start_state
        self.custom_start = start_state   # used when no book opening is selected
        try:
            self.book = load_book(book)["openings"]
        except FileNotFoundError:
            self.book = []                # no book generated yet: the Book selector is hidden
        self.opening = None               # index of the selected book opening, or None
        self.view = BoardView((BOARD_X, MARGIN), SQUARE)
        self.agent_factories, self.agent_errors = discover_agents()
        self.agents = [None, None]  # agent instance per color, or None for a human
        self.results = queue.Queue()  # (token, move, error) from agent threads
        self.token = 0  # increments whenever pending agent results become stale
        self.thinking = False
        # one lock per side: a stale search (e.g. after Undo) must finish before the same agent searches again,
        # since an agent's model and tree can't be used by two threads at once
        self.agent_locks = [threading.Lock(), threading.Lock()]
        self.paused = False
        self.message = None  # last error to show the user
        self.running = True

        # eval bar: one evaluator agent searching the current position on a background thread
        self.evaluator = None        # agent instance, or None when the bar is off
        self.eval_has_moves_left = False
        self.eval_token = 0          # increments whenever the position or the evaluator changes
        self.eval_started_for = None  # eval_token the last search was started for
        self.eval_running = False
        self.eval_info = None        # latest summarize() result for the current position
        self.eval_shown = None       # value drawn on the bar; eases toward eval_info["q"]
        self.move_sims = [MOVE_SIMS_DEFAULT, MOVE_SIMS_DEFAULT]   # simulations per move for each side's MCTS agent

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
        # a book opening starts from the normal position and plays its moves (they show in the move list and can be undone)
        self.start_state = STATE_INIT if self.opening is not None else self.custom_start
        self.game = State(self.start_state)
        self.snapshots = []  # (state, last_move, result) before each move, for undo
        self.move_labels = []
        self.last_move = None
        self.message = None
        self.result = self._initial_result()
        self.next_agent_time = 0.0
        self._invalidate_agent()
        self._invalidate_eval()
        self._clear_selection()
        if self.opening is not None:
            for move in self.book[self.opening]["moves"]:
                if not self.play(move):
                    break
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

    def set_opening(self, value):
        """Book selector: start a new game from that book opening (or the normal start for BOOK_OFF)."""
        self.opening = None if value == BOOK_OFF else self.opening_selector.index - 1
        self.new_game()

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
        self._invalidate_eval()
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
        self._invalidate_eval()
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
        agent, token, sims = self.agents[self.game.turn], self.token, self.move_sims[self.game.turn]
        state = copy.deepcopy(self.game)  # the agent may modify its copy as it searches
        lock = self.agent_locks[self.game.turn]

        def work():
            with lock:      # wait for a stale search on this agent to finish
                try:
                    if hasattr(agent, "analyze"):     # MCTS agent: search with the UI's simulation count
                        move = agent.analyze(state, sims, None)
                    else:
                        move = agent.get_move(state)
                    self.results.put((token, move, None))
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

    # ---------------------------------------------------------------- eval bar

    def set_evaluator(self, name):
        self._invalidate_eval()
        self.eval_shown = None
        self.evaluator = None
        self._resize(HEIGHT)
        if name == EVAL_OFF:
            return
        try:
            self.evaluator = self.agent_factories[name]()     # its own instance: never shared with a playing agent's thread
            self.eval_has_moves_left = len(self.evaluator.model.leaves) > 2
            self._resize(HEIGHT_EVAL)
        except Exception as e:
            traceback.print_exc()
            self.message = f"Could not create {name}: {type(e).__name__}: {e}"
            self.eval_selector.set_value(EVAL_OFF)

    def _resize(self, height):
        if height != self.height:
            self.height = height
            self.screen = pygame.display.set_mode((WIDTH, height))

    def _invalidate_eval(self):
        """The position (or evaluator) changed: a running search stops at its next progress report."""
        self.eval_token += 1
        self.eval_info = None

    def _maybe_start_eval(self):
        if self.evaluator is None or self.eval_running or self.eval_started_for == self.eval_token:
            return
        token = self.eval_token
        self.eval_started_for = token
        if self.result:                                   # game over: show the result, nothing to search
            q = terminal_value(str(self.result))
            self.eval_info = {"q": float(q or 0), "sims": 0, "depth": 0, "plies_left": 0.0 if q else None, "mate": None}
            return
        self.eval_running = True
        agent, state, has_moves_left = self.evaluator, copy.deepcopy(self.game), self.eval_has_moves_left
        last_sims = [0]
        lines = {"lines": [], "at": 0.0}                  # latest engine lines and when they were built

        def report(root, sims, force=False):
            info = summarize(root, sims, has_moves_left)
            now = time.monotonic()
            if force or now - lines["at"] >= ENGINE_LINES_EVERY:
                lines["lines"] = engine_lines(root, state, ENGINE_LINES, ENGINE_LINE_PLIES)
                lines["at"] = now
            info["lines"] = lines["lines"]
            if token == self.eval_token:
                self.eval_info = info

        def progress(root, sims):
            if token != self.eval_token:
                return False                              # position changed: stop searching
            last_sims[0] = sims
            report(root, sims)
            return True

        def work():
            try:
                agent.analyze(state, EVAL_MAX_SIMS, progress)     # keeps searching after a proven mate, for a shorter one
                # the search can end between progress reports (e.g. right after proving a mate): report once more
                root = getattr(agent.mcts, "root", None)
                if root is not None and token == self.eval_token:
                    report(root, last_sims[0], force=True)
            except Exception as e:
                traceback.print_exc()
                if token == self.eval_token:
                    self.message = f"Eval crashed: {type(e).__name__}: {e}"
            finally:
                self.eval_running = False

        threading.Thread(target=work, daemon=True).start()

    def _eval_text(self):
        if self.evaluator is None:
            return None
        info = self.eval_info
        if info is None:
            return "Evaluating..."
        q, mate = info["q"], info.get("mate")
        if mate:
            text = f"Mate in {abs(mate)} for {'White' if mate > 0 else 'Black'}"
        else:
            side = "even" if abs(q) < 0.005 else ("White" if q > 0 else "Black")
            text = f"Eval {q:+.2f} ({side})"
        if info["sims"]:
            text += f" - {info['sims']:,} sims, depth {info['depth']}"
        if info["plies_left"] is not None and not mate:
            text += f", ~{info['plies_left']:.0f} plies left"
        return text

    def set_move_sims(self, color, value):
        self.move_sims[color] = int(value.replace(",", ""))

    # ---------------------------------------------------------------- panel

    def _build_panel(self):
        x = BOARD_X + BOARD_PX + MARGIN
        w = PANEL_W - MARGIN
        half = (w - 10) // 2
        options = [HUMAN] + list(self.agent_factories)
        self.panel_x, self.panel_w = x, w
        # (selectors get their y positions from _layout, which runs every frame)
        self.selectors = [
            Selector((x + 62, 0, w - 62, 34), options,
                     lambda name, color=color: self.set_player(color, name))
            for color in (0, 1)
        ]
        # evaluators: agents that can analyze a position (MCTS agents)
        evaluators = [name for name, factory in self.agent_factories.items() if hasattr(factory, "analyze")]
        self.eval_selector = Selector((x + 62, 0, w - 62, 34), [EVAL_OFF] + evaluators, self.set_evaluator)
        # simulations per move, one per side, shown only while that side is an MCTS agent
        sims_options = [f"{n:,}" for n in MOVE_SIMS_OPTIONS]
        self.sims_selectors = [
            Selector((x + 62, 0, w - 62, 30), sims_options,
                     lambda value, color=color: self.set_move_sims(color, value),
                     index=MOVE_SIMS_OPTIONS.index(MOVE_SIMS_DEFAULT))
            for color in (0, 1)
        ]
        # book openings, labeled with their number and the reference search's eval (White's view)
        book_options = [BOOK_OFF] + [f"{i + 1}/{len(self.book)}  ({o['q']:+.2f})" for i, o in enumerate(self.book)]
        self.opening_selector = Selector((x + 62, 0, w - 62, 34), book_options, self.set_opening)
        self.buttons = [
            Button((x, 278, half, 36), "New game", self.new_game),
            Button((x + half + 10, 278, half, 36), "Undo", self.undo, lambda: bool(self.snapshots)),
            Button((x, 322, half, 36), lambda: "Resume" if self.paused else "Pause", self.toggle_pause,
                   lambda: self.paused or any(agent is not None for agent in self.agents)),
            Button((x + half + 10, 322, half, 36), "Flip board", self.flip),
        ]
        self.button_ys = [0, 0, 44, 44]     # offsets of the two button rows below the selectors

    def _shows_sims(self, color):
        """A side's Sims selector is shown while that side is an MCTS agent (one with analyze)."""
        return hasattr(self.agents[color], "analyze")

    def _panel_selectors(self):
        """(label, selector) rows currently shown in the panel, top to bottom."""
        rows = []
        for color in (0, 1):
            rows.append((COLOR_NAMES[color], self.selectors[color]))
            if self._shows_sims(color):
                rows.append(("Sims", self.sims_selectors[color]))
        rows.append(("Eval", self.eval_selector))
        if self.book:
            rows.append(("Book", self.opening_selector))
        return rows

    def _layout(self):
        """Stack the shown selector rows, then the buttons; sets self.content_y (where the status text starts)."""
        y = ROWS_TOP
        for _, selector in self._panel_selectors():
            sims_row = selector in self.sims_selectors
            if sims_row:
                y -= ROW_H - 34 - SIMS_GAP_ABOVE          # pull up under its player's selector (34 px tall)
                _place(selector, y)
                y += SIMS_ROW_H + SIMS_GAP_BELOW
            else:
                _place(selector, y)
                y += ROW_H
        buttons_y = y + 10
        for button, dy in zip(self.buttons, self.button_ys):
            button.rect.y = buttons_y + dy
        self.content_y = buttons_y + 98

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
        self._draw_eval_bar()
        self._draw_engine_lines()

        x, w = self.panel_x, self.panel_w
        self.screen.blit(self.title_font.render("Minichess", True, TEXT), (x, MARGIN - 4))
        self._layout()
        for name, selector in self._panel_selectors():
            sims_row = selector in self.sims_selectors
            font = self.small if sims_row else self.font
            label = font.render(name, True, TEXT_DIM if sims_row else TEXT)
            self.screen.blit(label, label.get_rect(midleft=(x + (14 if sims_row else 0), selector.rect.centery)))
            selector.draw(self.screen, font, mouse)
        for button in self.buttons:
            button.draw(self.screen, self.font, mouse)

        y = self.content_y
        for text, color in self._status_lines():
            for line in _wrap(text, self.font, w):
                self.screen.blit(self.font.render(line, True, color), (x, y))
                y += 21
        eval_text = self._eval_text()
        if eval_text:
            for line in _wrap(eval_text, self.small, w):
                self.screen.blit(self.small.render(line, True, TEXT_DIM), (x, y))
                y += 18
        if self.message:
            for line in _wrap(self.message, self.small, w):
                self.screen.blit(self.small.render(line, True, ERROR), (x, y))
                y += 18
        self._draw_moves(x, max(y + 14, self.content_y + 68), w)
        self._draw_footer(x, w)
        pygame.display.flip()

    def _draw_eval_bar(self):
        if self.evaluator is None:
            return
        target = self.eval_info["q"] if self.eval_info else None
        if target is None:
            shown = self.eval_shown
        elif self.eval_shown is None:
            shown = target
        else:
            shown = self.eval_shown + (target - self.eval_shown) * 0.2    # ease toward the new value
        self.eval_shown = shown
        mate = self.eval_info.get("mate") if self.eval_info else None
        draw_eval_bar(self.screen, (EVAL_X, MARGIN, EVAL_W, BOARD_PX), shown, self.view.flipped, self.small,
                      label=f"M{abs(mate)}" if mate else None)

    def _draw_engine_lines(self):
        if self.evaluator is None:
            return
        x, right = EVAL_X, BOARD_X + BOARD_PX
        lines = (self.eval_info or {}).get("lines") or []
        if not lines:
            text = "Searching..." if self.eval_info is None or not self.result else ""
            self.screen.blit(self.small.render(text, True, TEXT_DIM), (x, LINES_Y))
            return
        for k, line in enumerate(lines):
            y = LINES_Y + k * LINE_H
            mate = line["mate"]
            q = line["q"] if abs(line["q"]) >= 0.005 else 0.0      # no "-0.00"
            score = (f"{'+' if mate > 0 else '-'}M{abs(mate)}" if mate else f"{q:+.2f}")
            score_img = self.mono.render(f"{score:>6}", True, ACCENT if k == 0 else TEXT_DIM)
            self.screen.blit(score_img, (x, y))
            text_x = x + score_img.get_width() + 12
            text = line["text"]
            while text and self.mono.size(text)[0] > right - text_x:     # clip to the board's width
                text = text[:-2] + "\u2026"
            self.screen.blit(self.mono.render(text, True, TEXT if k == 0 else TEXT_DIM), (text_x, y))

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
        bottom = self.height - 52
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
        y = self.height - 14 - 18 * len(lines)
        for line in lines:
            self.screen.blit(self.small.render(line, True, ERROR if self.agent_errors else TEXT_DIM), (x, y))
            y += 18

    # ---------------------------------------------------------------- main loop

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._layout()
            for widget in [selector for _, selector in self._panel_selectors()] + self.buttons:
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
            self._maybe_start_eval()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


def _place(selector, y):
    """Move a Selector (and its arrow buttons) to the given top y."""
    selector.rect.y = selector.left.y = selector.right.y = y


def _normalize(move):
    """Accept lists, numpy ints, etc. and return ((row, col), (row, col)) of plain ints."""
    (r0, c0), (r1, c1) = move
    return ((int(r0), int(c0)), (int(r1), int(c1)))


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
