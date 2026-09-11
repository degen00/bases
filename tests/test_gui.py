"""Smoke tests for the pygame interface using SDL's dummy video driver."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame
    from bases.gui import App
except ImportError:  # pragma: no cover
    pygame = None


@unittest.skipIf(pygame is None, "pygame not installed")
class GuiTests(unittest.TestCase):
    def setUp(self):
        self.app = App(size=2, ai_delay_ms=0)

    def tearDown(self):
        pygame.quit()

    def click(self, pos):
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
        self.app.step()

    def test_menu_renders_and_starts_game(self):
        self.app.step()
        self.assertEqual(self.app.scene, "menu")
        play = next(b for b in self.app.buttons if b.label == "Play")
        self.click(play.rect.center)
        self.assertEqual(self.app.scene, "game")

    def test_human_click_draws_edge_and_ai_replies(self):
        self.app.mode = "hva"
        self.app.new_match()
        self.app.step()
        layout = self.app.board_layout()
        (ax, ay), (bx, by) = self.app.line_segment(0, layout)
        self.click(((ax + bx) // 2, (ay + by) // 2))
        self.assertTrue(self.app.board.is_drawn(0))
        self.assertEqual(self.app.board.turn, "B")
        self.app.step()  # schedules the AI
        self.app.step()  # AI moves (delay 0)
        self.assertEqual(len(self.app.board.history), 2)
        self.assertEqual(self.app.board.turn, "A")

    def test_undo_reverts_to_human_turn(self):
        self.app.mode = "hva"
        self.app.new_match()
        self.app.step()
        self.app.play(0)
        self.app.step(); self.app.step()
        self.assertEqual(len(self.app.board.history), 2)
        self.app.undo()
        self.assertEqual(len(self.app.board.history), 0)
        self.assertEqual(self.app.board.turn, "A")

    def test_ai_vs_ai_finishes_and_scores(self):
        self.app.mode = "ava"
        self.app.new_match()
        for _ in range(60):
            self.app.step()
            if self.app.scored:
                break
        self.assertTrue(self.app.board.is_full())
        self.assertEqual(sum(self.app.wins.values()), 1)

    def test_q_overlay_and_screenshot(self):
        self.app.mode = "hva"
        self.app.new_match()
        self.app.toggle_q()
        self.app.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shot.png"
            pygame.image.save(self.app.screen, str(path))
            self.assertGreater(path.stat().st_size, 1000)

    def test_hint_and_difficulty(self):
        self.app.mode = "hva"
        self.app.set_difficulty("hard")
        self.assertEqual(self.app.agent.search_depth, 3)
        self.app.new_match()
        self.app.step()
        self.app.show_hint()
        self.assertIn(self.app.hint, self.app.board.available())
        self.app.step()
        self.app.play(self.app.hint)
        self.assertIsNone(self.app.hint)
        self.app.set_difficulty("easy")
        self.assertGreater(self.app.agent.noise, 0)

    def test_keyboard_shortcuts(self):
        self.app.new_match()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.app.step()
        self.assertEqual(self.app.scene, "menu")
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        self.app.step()
        self.assertFalse(self.app.running)


if __name__ == "__main__":
    unittest.main()
