"""Kropki scene for the pygame interface (used by :mod:`bases.gui`)."""
from __future__ import annotations

import time
from typing import Optional

import pygame

from .kropki import KropkiBoard, Point, other
from .kropki_ai import make_kropki_ai

BG = (247, 247, 250)
PANEL = (255, 255, 255)
TEXT = (33, 37, 41)
MUTED = (125, 130, 140)
GRID = (222, 225, 231)
COLORS = {"A": (52, 120, 246), "B": (241, 143, 31)}
FILL = {"A": (52, 120, 246, 60), "B": (241, 143, 31, 60)}
GOOD = (46, 160, 90)
NEIGHBOURS8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


class KropkiView:
    """Board rendering, input and AI turns for one Kropki game."""

    def __init__(self, app):
        self.app = app
        self.board: Optional[KropkiBoard] = None
        self.ai = {}
        self.hover: Optional[Point] = None
        self.ai_due: Optional[float] = None
        self.scored = False
        self.message = ""

    # -- game control -----------------------------------------------------
    def start(self, width: int, height: int, mode: str, difficulty: str) -> None:
        self.board = KropkiBoard(width, height)
        self.mode = mode
        self.difficulty = difficulty
        self.ai = {p: make_kropki_ai(difficulty) for p, c in zip("AB", (mode[0], mode[2]))
                   if c == "a"}
        self.hover = None
        self.ai_due = None
        self.scored = False
        self.message = ""

    def is_ai(self, player: str) -> bool:
        return player in self.ai

    def player_label(self, player: str) -> str:
        return "AI" if self.is_ai(player) else "Human"

    def play(self, point: Optional[Point]) -> None:
        board = self.board
        if point is None:
            board.pass_turn()
        else:
            board.play(*point)
        self.hover = None
        self.ai_due = None

    def undo(self) -> None:
        board = self.board
        if board is None or not board.history or self.mode == "ava":
            return
        self.ai_due = None
        self.scored = False
        board.undo()
        while board.history and self.is_ai(board.turn):
            board.undo()
        self.message = ""

    def human_pass(self) -> None:
        board = self.board
        if board is not None and not board.is_over() and not self.is_ai(board.turn):
            self.play(None)

    def advance(self) -> None:
        board = self.board
        if board.is_over():
            if not self.scored:
                self.scored = True
                result = board.winner()
                self.app.wins[result] += 1
                a, b = board.scores["A"], board.scores["B"]
                self.message = ("It's a tie!" if result == "Tie"
                                else f"Player {result} wins {max(a, b)} - {min(a, b)}!")
            return
        if self.is_ai(board.turn):
            now = time.time()
            if self.ai_due is None:
                self.ai_due = now + self.app.ai_delay
            elif now >= self.ai_due:
                self.play(self.ai[board.turn].choose_move(board))

    # -- geometry ---------------------------------------------------------
    def layout(self):
        w, h = self.app.screen.get_size()
        margin = self.app.MARGIN
        avail_w = w - self.app.PANEL_W - 2 * margin
        avail_h = h - 2 * margin
        cols, rows = self.board.width, self.board.height
        cell = max(14, min(avail_w // cols, avail_h // rows, 60))
        ox = margin + (avail_w - cell * (cols - 1)) // 2
        oy = margin + (avail_h - cell * (rows - 1)) // 2
        return ox, oy, cell

    def screen_pos(self, x: int, y: int, layout) -> tuple[int, int]:
        ox, oy, cell = layout
        return ox + x * cell, oy + y * cell

    def point_at(self, pos, layout) -> Optional[Point]:
        ox, oy, cell = layout
        x = round((pos[0] - ox) / cell)
        y = round((pos[1] - oy) / cell)
        if not self.board.in_bounds(x, y):
            return None
        sx, sy = self.screen_pos(x, y, layout)
        if abs(pos[0] - sx) > cell * 0.45 or abs(pos[1] - sy) > cell * 0.45:
            return None
        return (x, y)

    # -- events -----------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.point_at(event.pos, self.layout())
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            board = self.board
            if board.is_over() or self.is_ai(board.turn):
                return
            point = self.point_at(event.pos, self.layout())
            if point is not None and board.is_playable(*point):
                self.play(point)
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_m):
                self.app.go_menu()
            elif event.key == pygame.K_n:
                self.app.new_match()
            elif event.key == pygame.K_u:
                self.undo()
            elif event.key == pygame.K_p:
                self.human_pass()

    # -- drawing ----------------------------------------------------------
    def draw(self) -> None:
        app, board = self.app, self.board
        screen = app.screen
        layout = self.layout()
        ox, oy, cell = layout
        cols, rows = board.width, board.height
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

        # territory fill
        half = cell // 2
        for y in range(rows):
            for x in range(cols):
                owner = board.territory[y][x]
                if owner:
                    sx, sy = self.screen_pos(x, y, layout)
                    pygame.draw.rect(overlay, FILL[owner],
                                     pygame.Rect(sx - half, sy - half, cell, cell))
        # grid
        for x in range(cols):
            pygame.draw.line(screen, GRID, self.screen_pos(x, 0, layout),
                             self.screen_pos(x, rows - 1, layout), 1)
        for y in range(rows):
            pygame.draw.line(screen, GRID, self.screen_pos(0, y, layout),
                             self.screen_pos(cols - 1, y, layout), 1)
        # base outlines: link owner dots that border a base region
        for base in board.bases:
            region = set(base.points)
            boundary = set()
            for (x, y) in region:
                for dx, dy in NEIGHBOURS8:
                    nx, ny = x + dx, y + dy
                    if board.in_bounds(nx, ny) and board.is_live(nx, ny, base.owner):
                        boundary.add((nx, ny))
            for (x, y) in boundary:
                for dx, dy in NEIGHBOURS8:
                    q = (x + dx, y + dy)
                    if q in boundary and q > (x, y):
                        pygame.draw.line(screen, COLORS[base.owner],
                                         self.screen_pos(x, y, layout),
                                         self.screen_pos(*q, layout), max(2, cell // 8))
        # hover ghost
        if (self.hover is not None and not board.is_over() and not self.is_ai(board.turn)
                and board.is_playable(*self.hover)):
            sx, sy = self.screen_pos(*self.hover, layout)
            pygame.draw.circle(overlay, (*COLORS[board.turn], 110), (sx, sy), max(4, cell // 3))
        screen.blit(overlay, (0, 0))
        # dots
        radius = max(3, cell // 4)
        last = board.last_move()
        for y in range(rows):
            for x in range(cols):
                d = board.dot[y][x]
                if d is None:
                    pygame.draw.circle(screen, GRID, self.screen_pos(x, y, layout), max(1, cell // 12))
                    continue
                sx, sy = self.screen_pos(x, y, layout)
                if board.is_live(x, y, d):
                    pygame.draw.circle(screen, COLORS[d], (sx, sy), radius)
                else:                                   # captured dot
                    pygame.draw.circle(screen, COLORS[d], (sx, sy), radius, 2)
                if last == (x, y):
                    pygame.draw.circle(screen, (255, 255, 255), (sx, sy), max(2, radius // 2))
        self.draw_panel()

    def draw_panel(self) -> None:
        app, board = self.app, self.board
        screen = app.screen
        w, h = screen.get_size()
        px = w - app.PANEL_W
        pygame.draw.rect(screen, PANEL, pygame.Rect(px, 0, app.PANEL_W, h))
        pygame.draw.line(screen, GRID, (px, 0), (px, h))
        x, inner, y = px + 20, app.PANEL_W - 40, 24
        header = f"Kropki {board.width} x {board.height}"
        if self.ai:
            header += f"  -  AI {self.difficulty}"
        app.text(header, (x, y), app.small, MUTED)
        y += 30
        for player in ("A", "B"):
            card = pygame.Rect(x, y, inner, 56)
            active = board.turn == player and not board.is_over()
            pygame.draw.rect(screen, BG, card, border_radius=8)
            if active:
                pygame.draw.rect(screen, COLORS[player], card, 2, border_radius=8)
            pygame.draw.circle(screen, COLORS[player], (x + 14, y + 28), 8)
            app.text(f"Player {player}", (x + 30, y + 8))
            app.text(self.player_label(player), (x + 30, y + 32), app.small, MUTED)
            app.text(str(board.scores[player]), (x + inner - 12, y + 12), app.big, right=True)
            y += 66
        y += 6
        if board.is_over():
            msg, color = self.message, GOOD
        elif self.is_ai(board.turn):
            msg, color = "AI is thinking...", MUTED
        else:
            msg, color = f"Player {board.turn}: place a dot", COLORS[board.turn]
        app.text(msg, (x, y), color=color)
        y += 30
        if board.passes == 1 and not board.is_over():
            app.text(f"{other(board.turn)} passed - pass too to end the game",
                     (x, y), app.small, MUTED)
        y += 26
        app.text(f"Match: A {app.wins['A']}  -  B {app.wins['B']}  (ties {app.wins['Tie']})",
                 (x, y), app.small, MUTED)
        y += 34
        human_turn = not board.is_over() and not self.is_ai(board.turn)
        from .gui import Button
        app.button_row([Button("New game", app.new_match),
                        Button("Undo", self.undo, enabled=bool(board.history)
                               and self.mode != "ava"),
                        Button("Pass", self.human_pass, enabled=human_turn)], y, x, inner)
        y += 46
        app.button_row([Button("Menu", app.go_menu)], y, x, inner)
        y += 56
        for line in ("Enclose enemy dots with a closed chain of",
                     "your dots (diagonals count) to capture them.",
                     "Chains around empty points are houses: a dot",
                     "placed inside is captured at once."):
            app.text(line, (x, y), app.small, MUTED)
            y += 20
        app.text("N new  U undo  P pass  Esc menu", (x, h - 28), app.small, MUTED)
