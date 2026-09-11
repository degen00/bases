"""Pygame graphical interface: play, watch the AI, and train it.

Run with ``python play.py`` (or ``python play.py gui --size 4``).

Keys: click an edge to draw it | N new game | U undo | H hint | Q show the
AI's Q-values | Esc back to the menu.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional

import pygame

from .agent import QLearningAgent
from .board import Board
from .config import Config, load_config
from .train import TrainingStats, train_agents

# -- palette ------------------------------------------------------------------
BG = (247, 247, 250)
PANEL = (255, 255, 255)
TEXT = (33, 37, 41)
MUTED = (125, 130, 140)
DOT = (45, 48, 55)
GRID = (222, 225, 231)
COLORS = {"A": (52, 120, 246), "B": (241, 143, 31)}
LIGHT = {"A": (52, 120, 246, 70), "B": (241, 143, 31, 70)}
BTN = (236, 238, 243)
BTN_HOVER = (222, 226, 235)
BTN_ACTIVE = (52, 120, 246)
GOOD = (46, 160, 90)
BAD = (220, 70, 60)

MODES = [("hvh", "Human vs Human"), ("hva", "Human vs AI"),
         ("avh", "AI vs Human"), ("ava", "AI vs AI")]
# difficulty -> (probability of a random move, search depth for boards <= 3x3,
# search depth for larger boards)
DIFFICULTIES = {"easy": (0.35, 0, 0), "normal": (0.0, 0, 0), "hard": (0.0, 3, 2)}
HINT_COLOR = (46, 160, 90)
MIN_SIZE, MAX_SIZE = 1, 8
EPISODE_STEPS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]


class Button:
    def __init__(self, label: str, callback: Callable[[], None], *,
                 active: bool = False, enabled: bool = True):
        self.label = label
        self.callback = callback
        self.active = active
        self.enabled = enabled
        self.rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, mouse) -> None:
        hovered = self.enabled and self.rect.collidepoint(mouse)
        color = BTN_ACTIVE if self.active else BTN_HOVER if hovered else BTN
        pygame.draw.rect(surf, color, self.rect, border_radius=8)
        fg = (255, 255, 255) if self.active else TEXT if self.enabled else MUTED
        txt = font.render(self.label, True, fg)
        if txt.get_width() > self.rect.width - 12:
            txt = pygame.font.Font(None, 20).render(self.label, True, fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def click(self, pos) -> bool:
        if self.enabled and self.rect.collidepoint(pos):
            self.callback()
            return True
        return False


class App:
    """The whole interface; :meth:`run` loops :meth:`step` until quit."""

    PANEL_W = 300
    MARGIN = 28

    def __init__(self, size: int = 3, extra_turn_on_box: bool = False,
                 config: Optional[Config] = None, ai_delay_ms: int = 350,
                 window=(1000, 660)):
        pygame.init()
        pygame.display.set_caption("Bases")
        self.screen = pygame.display.set_mode(window, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 26)
        self.small = pygame.font.Font(None, 20)
        self.big = pygame.font.Font(None, 48)
        self.cfg = config or load_config()
        self.size = max(MIN_SIZE, min(MAX_SIZE, size))
        self.extra_turn = extra_turn_on_box
        self.mode = "hva"
        self.difficulty = "normal"
        self.ai_delay = ai_delay_ms / 1000.0
        self.scene = "menu"
        self.running = True
        self.buttons: list[Button] = []
        self.message = ""
        # game state
        self.board: Optional[Board] = None
        self.hover: Optional[int] = None
        self.ai_due: Optional[float] = None
        self.show_q = False
        self.hint: Optional[int] = None
        self.scored = False
        self.wins = {"A": 0, "B": 0, "Tie": 0}
        # AI
        self.agent: Optional[QLearningAgent] = None
        self.agent_status = ""
        self.load_agent()
        # training
        self.train_episodes = 20_000
        self.train_thread: Optional[threading.Thread] = None
        self.train_stats: Optional[TrainingStats] = None
        self.train_stop = False
        self.train_error: Optional[str] = None
        self.train_result: Optional[QLearningAgent] = None
        self.train_started = 0.0

    # -- AI management ------------------------------------------------------
    def load_agent(self) -> None:
        path = self.cfg.path("policy", self.size)
        agent = QLearningAgent(self.size, training_mode=False,
                               extra_turn_on_box=self.extra_turn)
        status = "AI untrained: plays the greedy heuristic"
        if path.is_file():
            try:
                agent.load_policy(path)
                status = (f"AI policy: {agent.num_states:,} states, "
                          f"{agent.episodes_trained:,} training games")
                if agent.extra_turn_on_box != self.extra_turn:
                    status += " (trained with the other turn rule)"
            except (ValueError, OSError, KeyError) as exc:
                status = f"AI policy unreadable ({exc.__class__.__name__}); using greedy"
        self.agent = agent
        self.agent_status = status
        self.apply_difficulty()

    def hard_depth(self) -> int:
        return DIFFICULTIES["hard"][1 if self.size <= 3 else 2]

    def apply_difficulty(self) -> None:
        noise, small, large = DIFFICULTIES[self.difficulty]
        self.agent.noise = noise
        self.agent.search_depth = small if self.size <= 3 else large

    def set_difficulty(self, level: str) -> None:
        self.difficulty = level
        self.apply_difficulty()

    def show_hint(self) -> None:
        """Highlight the move the strongest AI setting would play now."""
        board = self.board
        if board is None or board.is_full() or self.is_ai(board.turn):
            return
        values = self.agent.action_values(board, depth=self.hard_depth())
        best = max(values.values())
        candidates = [a for a, v in values.items() if v == best]
        self.hint = self.agent._greedy(board, candidates)

    def is_ai(self, player: str) -> bool:
        return self.mode[0 if player == "A" else 2] == "a"

    def player_label(self, player: str) -> str:
        return "AI" if self.is_ai(player) else "Human"

    # -- scene changes ------------------------------------------------------
    def start_game(self) -> None:
        self.board = Board(self.size, extra_turn_on_box=self.extra_turn)
        self.hover = None
        self.hint = None
        self.ai_due = None
        self.scored = False
        self.message = ""
        self.scene = "game"

    def go_menu(self) -> None:
        self.scene = "menu"
        self.message = ""

    def new_match(self) -> None:
        self.wins = {"A": 0, "B": 0, "Tie": 0}
        self.start_game()

    def undo(self) -> None:
        board = self.board
        if board is None or not board.history or self.mode == "ava":
            return
        self.ai_due = None
        self.scored = False
        board.undo()
        # Keep undoing until a human is to move again.
        while board.history and self.is_ai(board.turn):
            board.undo()
        self.hint = None
        self.message = ""

    def toggle_q(self) -> None:
        self.show_q = not self.show_q

    def change_size(self, delta: int) -> None:
        new = max(MIN_SIZE, min(MAX_SIZE, self.size + delta))
        if new != self.size:
            self.size = new
            self.load_agent()

    def set_rule(self, extra_turn: bool) -> None:
        if extra_turn != self.extra_turn:
            self.extra_turn = extra_turn
            self.load_agent()

    def change_episodes(self, delta: int) -> None:
        i = min(range(len(EPISODE_STEPS)),
                key=lambda j: abs(EPISODE_STEPS[j] - self.train_episodes))
        self.train_episodes = EPISODE_STEPS[max(0, min(len(EPISODE_STEPS) - 1, i + delta))]

    # -- training -------------------------------------------------------------
    def start_training(self) -> None:
        if self.train_thread is not None and self.train_thread.is_alive():
            return
        self.train_stop = False
        self.train_stats = None
        self.train_error = None
        self.train_result = None
        self.train_started = time.time()
        self.scene = "train"
        size, episodes, extra = self.size, self.train_episodes, self.extra_turn
        # Continue from the saved policy (a fresh copy, so the agent used for
        # play is never touched from the training thread).
        resume = None
        path = self.cfg.path("policy", size)
        if (self.agent and self.agent.num_states and path.is_file()
                and self.agent.extra_turn_on_box == extra):
            try:
                resume = QLearningAgent.load(path, training_mode=True)
            except (ValueError, OSError, KeyError):
                resume = None

        def progress(stats: TrainingStats):
            self.train_stats = stats
            return not self.train_stop

        def worker():
            try:
                self.train_result = train_agents(
                    size, episodes, extra_turn_on_box=extra, progress=progress,
                    eval_every=max(1, episodes // 20), eval_episodes=100,
                    agent=resume, config=self.cfg, quiet=True)
            except Exception as exc:  # surfaced in the UI
                self.train_error = f"{exc.__class__.__name__}: {exc}"

        self.train_thread = threading.Thread(target=worker, daemon=True)
        self.train_thread.start()

    def stop_training(self) -> None:
        self.train_stop = True

    def poll_training(self) -> None:
        t = self.train_thread
        if t is None or t.is_alive():
            return
        self.train_thread = None
        if self.train_error:
            self.message = f"Training failed: {self.train_error}"
        else:
            self.load_agent()
            self.message = "Training finished. " + self.agent_status
        self.scene = "menu"

    # -- geometry -------------------------------------------------------------
    def board_layout(self):
        w, h = self.screen.get_size()
        n = self.size
        avail_w = w - self.PANEL_W - 2 * self.MARGIN
        avail_h = h - 2 * self.MARGIN
        cell = max(24, min(avail_w, avail_h) // n)
        cell = min(cell, 140)
        side = cell * n
        ox = self.MARGIN + (avail_w - side) // 2
        oy = self.MARGIN + (avail_h - side) // 2
        return ox, oy, cell

    def dot_pos(self, x: int, y: int, layout) -> tuple[int, int]:
        ox, oy, cell = layout
        return ox + x * cell, oy + y * cell

    def line_segment(self, idx: int, layout):
        x1, y1, x2, y2 = self.board.geom.coords[idx]
        return self.dot_pos(x1, y1, layout), self.dot_pos(x2, y2, layout)

    def edge_at(self, pos, layout) -> Optional[int]:
        """Nearest undrawn edge to ``pos`` within a tolerance, else None."""
        if self.board is None:
            return None
        cell = layout[2]
        best, best_d = None, cell * 0.28
        px, py = pos
        for idx in self.board.available():
            (ax, ay), (bx, by) = self.line_segment(idx, layout)
            # distance from point to segment
            dx, dy = bx - ax, by - ay
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
            d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d < best_d:
                best, best_d = idx, d
        return best

    # -- main loop --------------------------------------------------------------
    def run(self) -> None:
        while self.running:
            self.step()
        pygame.quit()

    def step(self) -> None:
        """Handle events (against last frame's buttons), advance the AI, draw."""
        for event in pygame.event.get():
            self.handle_event(event)
        if self.scene == "game":
            self.advance_game()
        elif self.scene == "train":
            self.poll_training()
        self.draw()
        pygame.display.flip()
        self.clock.tick(60)

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode(
                (max(640, event.w), max(480, event.h)), pygame.RESIZABLE)
        elif event.type == pygame.KEYDOWN:
            self.handle_key(event.key)
        elif event.type == pygame.MOUSEMOTION and self.scene == "game":
            self.hover = self.edge_at(event.pos, self.board_layout())
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.click(event.pos):
                    return
            if self.scene == "game":
                self.handle_board_click(event.pos)

    def handle_key(self, key: int) -> None:
        if self.scene == "menu":
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self.new_match()
            elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.change_size(1)
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.change_size(-1)
        elif self.scene == "game":
            if key in (pygame.K_ESCAPE, pygame.K_m):
                self.go_menu()
            elif key == pygame.K_n:
                self.start_game()
            elif key == pygame.K_u:
                self.undo()
            elif key == pygame.K_q:
                self.toggle_q()
            elif key == pygame.K_h:
                self.show_hint()
        elif self.scene == "train":
            if key == pygame.K_ESCAPE:
                self.stop_training()

    def handle_board_click(self, pos) -> None:
        board = self.board
        if board is None or board.is_full() or self.is_ai(board.turn):
            return
        idx = self.edge_at(pos, self.board_layout())
        if idx is not None:
            self.play(idx)

    def play(self, idx: int) -> None:
        self.board.play_index(idx)
        self.hover = None
        self.hint = None
        self.ai_due = None

    def advance_game(self) -> None:
        board = self.board
        if board.is_full():
            if not self.scored:
                self.scored = True
                result = board.winner()
                self.wins[result] += 1
                a, b = board.scores["A"], board.scores["B"]
                self.message = ("It's a tie!" if result == "Tie"
                                else f"Player {result} wins {max(a, b)} - {min(a, b)}!")
            return
        if self.is_ai(board.turn):
            now = time.time()
            if self.ai_due is None:
                self.ai_due = now + self.ai_delay
            elif now >= self.ai_due:
                self.play(self.agent.choose_index(board))
                self.hover = None

    # -- drawing ----------------------------------------------------------------
    def text(self, s: str, pos, font=None, color=TEXT, center=False, right=False):
        surf = (font or self.font).render(s, True, color)
        rect = surf.get_rect()
        if center:
            rect.center = pos
        elif right:
            rect.topright = pos
        else:
            rect.topleft = pos
        self.screen.blit(surf, rect)
        return rect

    def button_row(self, specs, y: int, x: int, width: int, height: int = 36, gap: int = 8):
        """Lay out buttons evenly across ``width`` starting at ``x``."""
        if not specs:
            return
        bw = (width - gap * (len(specs) - 1)) // len(specs)
        for i, button in enumerate(specs):
            button.rect = pygame.Rect(x + i * (bw + gap), y, bw, height)
            self.buttons.append(button)

    def draw(self) -> None:
        self.buttons = []
        self.screen.fill(BG)
        if self.scene == "menu":
            self.draw_menu()
        elif self.scene == "game":
            self.draw_game()
        else:
            self.draw_train()
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(self.screen, self.font, mouse)

    def draw_menu(self) -> None:
        w, h = self.screen.get_size()
        cx = w // 2
        col_w = min(560, w - 2 * self.MARGIN)
        x = cx - col_w // 2
        self.text("Bases", (cx, 70), self.big, center=True)
        self.text("Dots and Boxes against a Q-learning opponent", (cx, 108),
                  self.small, MUTED, center=True)

        y = 160
        self.text("Board", (x, y + 8))
        self.button_row([Button("-", lambda: self.change_size(-1)),
                         Button(f"{self.size} x {self.size} boxes", lambda: None, active=True),
                         Button("+", lambda: self.change_size(1))], y, x + 90, col_w - 90)
        y += 56
        self.text("Rule", (x, y + 8))
        self.button_row([Button("Alternate turns", lambda: self.set_rule(False),
                                active=not self.extra_turn),
                         Button("Extra turn after a box", lambda: self.set_rule(True),
                                active=self.extra_turn)], y, x + 90, col_w - 90)
        y += 56
        self.text("Players", (x, y + 8))
        self.button_row([Button(label, lambda m=key: setattr(self, "mode", m),
                                active=self.mode == key) for key, label in MODES],
                        y, x + 90, col_w - 90)
        y += 56
        self.text("AI level", (x, y + 8))
        self.button_row([Button(level.capitalize(), lambda l=level: self.set_difficulty(l),
                                active=self.difficulty == level) for level in DIFFICULTIES],
                        y, x + 90, col_w - 90)
        y += 64
        status_color = GOOD if self.agent and self.agent.num_states else MUTED
        self.text(self.agent_status, (cx, y), self.small, status_color, center=True)
        y += 36
        self.text("Training", (x, y + 8))
        self.button_row([Button("-", lambda: self.change_episodes(-1)),
                         Button(f"{self.train_episodes:,} games", lambda: None, active=True),
                         Button("+", lambda: self.change_episodes(1)),
                         Button("Train AI", self.start_training)], y, x + 90, col_w - 90)
        y += 72
        play = Button("Play", self.new_match, active=True)
        play.rect = pygame.Rect(cx - 120, y, 240, 48)
        self.buttons.append(play)
        y += 70
        if self.message:
            self.text(self.message, (cx, y), self.small, TEXT, center=True)
        self.text("Enter: play   +/-: board size   Esc: quit", (cx, h - 24),
                  self.small, MUTED, center=True)

    def draw_game(self) -> None:
        board = self.board
        layout = self.board_layout()
        ox, oy, cell = layout
        n = self.size
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        # box fills
        for r in range(n):
            for c in range(n):
                owner = board.owner[r * n + c]
                if owner:
                    rect = pygame.Rect(ox + c * cell, oy + r * cell, cell, cell)
                    pygame.draw.rect(overlay, LIGHT[owner], rect.inflate(-6, -6), border_radius=6)
                    label = self.big.render(owner, True, COLORS[owner])
                    label.set_alpha(140)
                    overlay.blit(label, label.get_rect(center=rect.center))
        # faint undrawn edges
        for idx in board.available():
            a, b = self.line_segment(idx, layout)
            pygame.draw.line(self.screen, GRID, a, b, 2)
        # Q-value overlay
        q_values = None
        if self.show_q and self.agent is not None and not board.is_full():
            q_values = self.agent.q_values(board)
            vmax = max(1e-6, max(abs(v) for v in q_values.values()))
            for idx, v in q_values.items():
                a, b = self.line_segment(idx, layout)
                k = min(1.0, abs(v) / vmax)
                base = GOOD if v > 0 else BAD if v < 0 else MUTED
                color = (*base, int(40 + 180 * k))
                pygame.draw.line(overlay, color, a, b, max(3, cell // 14))
                if n <= 5:
                    mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                    lbl = self.small.render(f"{v:+.2f}", True, TEXT)
                    bg = lbl.get_rect(center=mid).inflate(4, 2)
                    pygame.draw.rect(overlay, (255, 255, 255, 200), bg, border_radius=3)
                    overlay.blit(lbl, lbl.get_rect(center=mid))
        # drawn lines
        width = max(4, cell // 9)
        last = board.history[-1][0] if board.history else None
        for idx, player, _ in board.history:
            a, b = self.line_segment(idx, layout)
            pygame.draw.line(self.screen, COLORS[player], a, b, width)
        if last is not None:
            a, b = self.line_segment(last, layout)
            pygame.draw.line(overlay, (255, 255, 255, 110), a, b, max(2, width // 3))
        # hint
        if self.hint is not None and not board.is_drawn(self.hint):
            a, b = self.line_segment(self.hint, layout)
            pygame.draw.line(overlay, (*HINT_COLOR, 170), a, b, width)
            pygame.draw.circle(overlay, (*HINT_COLOR, 220),
                               ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2), max(5, width))
        # hover
        if (self.hover is not None and not board.is_full() and not self.is_ai(board.turn)
                and not board.is_drawn(self.hover)):
            a, b = self.line_segment(self.hover, layout)
            pygame.draw.line(overlay, (*COLORS[board.turn], 120), a, b, width)
        self.screen.blit(overlay, (0, 0))
        # dots
        radius = max(4, cell // 12)
        for r in range(n + 1):
            for c in range(n + 1):
                pygame.draw.circle(self.screen, DOT, self.dot_pos(c, r, layout), radius)

        self.draw_panel(q_values)

    def draw_panel(self, q_values) -> None:
        board = self.board
        w, h = self.screen.get_size()
        px = w - self.PANEL_W
        panel = pygame.Rect(px, 0, self.PANEL_W, h)
        pygame.draw.rect(self.screen, PANEL, panel)
        pygame.draw.line(self.screen, GRID, (px, 0), (px, h))
        x = px + 20
        inner = self.PANEL_W - 40
        y = 24
        rule = "extra turn after a box" if self.extra_turn else "alternate turns"
        header = f"{self.size} x {self.size}  -  {rule}"
        if "a" in self.mode:
            header += f"  -  AI {self.difficulty}"
        self.text(header, (x, y), self.small, MUTED)
        y += 30
        for player in ("A", "B"):
            card = pygame.Rect(x, y, inner, 56)
            active = board.turn == player and not board.is_full()
            pygame.draw.rect(self.screen, BG, card, border_radius=8)
            if active:
                pygame.draw.rect(self.screen, COLORS[player], card, 2, border_radius=8)
            pygame.draw.rect(self.screen, COLORS[player], pygame.Rect(x + 8, y + 10, 10, 36),
                             border_radius=4)
            self.text(f"Player {player}", (x + 28, y + 8))
            self.text(self.player_label(player), (x + 28, y + 32), self.small, MUTED)
            self.text(str(board.scores[player]), (x + inner - 12, y + 12), self.big, right=True)
            y += 66
        y += 6
        if board.is_full():
            msg, color = self.message, GOOD
        elif self.is_ai(board.turn):
            msg, color = "AI is thinking...", MUTED
        else:
            msg, color = f"Player {board.turn}: click an edge", COLORS[board.turn]
        self.text(msg, (x, y), color=color)
        y += 34
        self.text(f"Match: A {self.wins['A']}  -  B {self.wins['B']}  "
                  f"(ties {self.wins['Tie']})", (x, y), self.small, MUTED)
        y += 34

        human_turn = not board.is_full() and not self.is_ai(board.turn)
        self.button_row([Button("New game", self.start_game),
                         Button("Undo", self.undo, enabled=bool(board.history)
                                and self.mode != "ava"),
                         Button("Hint", self.show_hint, enabled=human_turn)], y, x, inner)
        y += 46
        has_ai = "a" in self.mode
        self.button_row([Button("Q-values", self.toggle_q, active=self.show_q,
                                enabled=has_ai),
                         Button("Menu", self.go_menu)], y, x, inner)
        y += 56

        if self.show_q and has_ai:
            if q_values is None:
                note = "Game over."
            elif not self.agent.knows(board):
                note = "Position not learned yet: values shown are the heuristic prior."
            else:
                best = max(q_values.values())
                note = f"Learned value: {best:+.2f} boxes for the mover."
                if self.hover in q_values:
                    note += f"  Hovered edge: {q_values[self.hover]:+.2f}"
            for line in wrap(note, self.small, inner):
                self.text(line, (x, y), self.small, MUTED)
                y += 20
            y += 8
        if has_ai:
            for line in wrap(self.agent_status, self.small, inner):
                self.text(line, (x, y), self.small, MUTED)
                y += 20
        self.text("N new  U undo  H hint  Q values  Esc menu", (x, h - 28), self.small, MUTED)

    def draw_train(self) -> None:
        w, h = self.screen.get_size()
        cx = w // 2
        col_w = min(600, w - 2 * self.MARGIN)
        x = cx - col_w // 2
        self.text(f"Training the {self.size} x {self.size} AI", (cx, 70), self.big, center=True)
        stats = self.train_stats
        frac = stats.fraction if stats else 0.0
        episode = stats.episode if stats else 0
        y = 130
        pygame.draw.rect(self.screen, BTN, pygame.Rect(x, y, col_w, 22), border_radius=11)
        if frac > 0:
            pygame.draw.rect(self.screen, BTN_ACTIVE,
                             pygame.Rect(x, y, int(col_w * frac), 22), border_radius=11)
        elapsed = time.time() - self.train_started
        eta = (elapsed / frac - elapsed) if frac > 0.02 else None
        self.text(f"{episode:,} / {self.train_episodes:,} games   "
                  f"{elapsed:,.0f}s elapsed" + (f",  about {eta:,.0f}s left" if eta else ""),
                  (cx, y + 44), self.small, MUTED, center=True)
        y += 90
        if stats is not None:
            rows = [("States in Q-table", f"{stats.states:,}"),
                    ("Exploration rate", f"{stats.exploration_rate:.3f}")]
            if stats.win_rate_random is not None:
                rows += [("Win rate vs random", f"{stats.win_rate_random:.0%}"),
                         ("Win rate vs greedy", f"{stats.win_rate_greedy:.0%}"),
                         ("Avg margin vs greedy", f"{stats.avg_margin_greedy:+.2f} boxes")]
            for label, value in rows:
                self.text(label, (x, y), color=MUTED)
                self.text(value, (x + col_w, y), right=True)
                y += 32
        else:
            self.text("Warming up...", (cx, y), color=MUTED, center=True)
            y += 32
        y += 20
        stop = Button("Stop early (keeps progress)" if not self.train_stop
                      else "Stopping at next checkpoint...", self.stop_training,
                      enabled=not self.train_stop)
        stop.rect = pygame.Rect(cx - 160, y, 320, 44)
        self.buttons.append(stop)


def wrap(text: str, font: pygame.font.Font, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if font.size(trial)[0] <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def run(size: int = 3, extra_turn_on_box: bool = False,
        config: Optional[Config] = None, difficulty: str = "normal") -> None:
    app = App(size=size, extra_turn_on_box=extra_turn_on_box, config=config)
    app.set_difficulty(difficulty)
    app.run()


if __name__ == "__main__":
    run()
