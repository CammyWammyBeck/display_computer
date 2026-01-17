"""Training mode renderer - shows graphs and stats instead of game."""

import time
import pygame

from shared.config import config as global_config
from shared.graphs import LineGraph, GraphStyle

from .config import flappy_config as cfg


class TrainingRenderer:
    """
    Renders training progress with graphs and statistics.

    Replaces the game renderer during training mode for faster learning
    while still providing visual feedback.
    """

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.start_time = time.time()
        self.last_episode_time = time.time()
        self.episodes_this_second = 0
        self.eps_per_second = 0.0

        # Fonts
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)
        self.font_title = pygame.font.Font(None, 64)

        # Graph styles
        score_style = GraphStyle(
            line_color=(100, 255, 150),
            fill_color=(100, 255, 150, 40),
        )
        avg_style = GraphStyle(
            line_color=(100, 200, 255),
            fill_color=(100, 200, 255, 40),
        )
        epsilon_style = GraphStyle(
            line_color=(255, 200, 100),
            fill_color=(255, 200, 100, 40),
            show_fill=True,
        )
        loss_style = GraphStyle(
            line_color=(255, 100, 150),
            fill_color=(255, 100, 150, 40),
        )

        # Initialize graphs
        self.score_graph = LineGraph(
            max_points=cfg.graph_history_length,
            title="Episode Scores",
            style=score_style,
            y_min=0,
        )
        self.avg_graph = LineGraph(
            max_points=cfg.graph_history_length,
            title="Rolling Average (100 episodes)",
            style=avg_style,
            y_min=0,
        )
        self.epsilon_graph = LineGraph(
            max_points=cfg.graph_history_length,
            title="Exploration Rate (Epsilon)",
            style=epsilon_style,
            y_min=0,
            y_max=1,
        )
        self.loss_graph = LineGraph(
            max_points=cfg.graph_history_length,
            title="Training Loss",
            style=loss_style,
            y_min=0,
        )

        # Mini game preview
        self.preview_surface = pygame.Surface((300, 400))
        self.show_preview = False
        self.preview_game_state = None

        # Layout calculations
        self._calculate_layout()

    def _calculate_layout(self):
        """Calculate positions for all UI elements."""
        w = global_config.screen_width
        h = global_config.screen_height

        # Graphs take left 2/3 of screen
        graph_width = int(w * 0.65)
        graph_height = (h - 150) // 2 - 20

        padding = 20
        top_y = 100

        # Graph rectangles
        self.score_graph_rect = pygame.Rect(
            padding, top_y,
            graph_width // 2 - padding, graph_height
        )
        self.avg_graph_rect = pygame.Rect(
            graph_width // 2 + padding, top_y,
            graph_width // 2 - padding, graph_height
        )
        self.epsilon_graph_rect = pygame.Rect(
            padding, top_y + graph_height + 20,
            graph_width // 2 - padding, graph_height
        )
        self.loss_graph_rect = pygame.Rect(
            graph_width // 2 + padding, top_y + graph_height + 20,
            graph_width // 2 - padding, graph_height
        )

        # Stats panel on right
        self.stats_x = graph_width + 40
        self.stats_y = top_y
        self.stats_width = w - graph_width - 60

        # Preview below stats
        self.preview_x = self.stats_x + (self.stats_width - 300) // 2
        self.preview_y = h - 450

    def update_episode_rate(self):
        """Track episodes per second."""
        self.episodes_this_second += 1

        current_time = time.time()
        if current_time - self.last_episode_time >= 1.0:
            self.eps_per_second = self.episodes_this_second
            self.episodes_this_second = 0
            self.last_episode_time = current_time

    def add_episode_data(
        self,
        score: int,
        avg_score: float,
        epsilon: float,
        loss: float
    ):
        """Add data from a completed episode."""
        self.score_graph.add_point(score)
        self.avg_graph.add_point(avg_score)
        self.epsilon_graph.add_point(epsilon)
        if loss > 0:
            self.loss_graph.add_point(loss)
        self.update_episode_rate()

    def load_history(self, history: dict):
        """Load training history into graphs."""
        # Clear existing data
        self.score_graph.clear()
        self.avg_graph.clear()
        self.epsilon_graph.clear()
        self.loss_graph.clear()

        # Load history
        scores = history.get("scores", [])
        averages = history.get("averages", [])
        epsilons = history.get("epsilons", [])
        losses = history.get("losses", [])

        for score in scores:
            self.score_graph.add_point(score)
        for avg in averages:
            self.avg_graph.add_point(avg)
        for eps in epsilons:
            self.epsilon_graph.add_point(eps)
        for loss in losses:
            if loss > 0:
                self.loss_graph.add_point(loss)

        print(f"Loaded {len(scores)} episodes of history into graphs")

    def set_preview_state(self, game_state: dict):
        """Set the game state for mini preview."""
        self.preview_game_state = game_state
        self.show_preview = True

    def hide_preview(self):
        """Hide the mini preview."""
        self.show_preview = False

    def render(self, agent_stats: dict, current_score: int = 0):
        """Render the training visualization."""
        # Draw title
        self._draw_title()

        # Draw graphs
        self.score_graph.render(self.screen, self.score_graph_rect)
        self.avg_graph.render(self.screen, self.avg_graph_rect)
        self.epsilon_graph.render(self.screen, self.epsilon_graph_rect)
        self.loss_graph.render(self.screen, self.loss_graph_rect)

        # Draw stats panel
        self._draw_stats(agent_stats)

        # Draw mini preview if active
        if self.show_preview and self.preview_game_state:
            self._draw_preview()

        # Draw controls hint
        self._draw_controls_hint()

    def _draw_title(self):
        """Draw the title bar."""
        title = self.font_title.render(
            "FLAPPY RL - TRAINING MODE",
            True,
            global_config.primary_color
        )
        self.screen.blit(title, (20, 20))

        # Training speed indicator
        speed_text = self.font_medium.render(
            f"Speed: {cfg.episodes_per_frame}x",
            True,
            global_config.warning_color
        )
        self.screen.blit(
            speed_text,
            (global_config.screen_width - speed_text.get_width() - 20, 30)
        )

    def _draw_stats(self, agent_stats: dict):
        """Draw the statistics panel."""
        x = self.stats_x
        y = self.stats_y

        # Panel background
        panel_rect = pygame.Rect(
            x - 15, y - 10,
            self.stats_width, 320
        )
        pygame.draw.rect(
            self.screen,
            (30, 30, 45),
            panel_rect,
            border_radius=10
        )

        # Title
        title = self.font_medium.render("STATISTICS", True, (180, 180, 180))
        self.screen.blit(title, (x, y))
        y += 45

        # Stats
        stats = [
            ("Episode", f"{agent_stats.get('episode', 0):,}"),
            ("Best Score", str(agent_stats.get('best_score', 0))),
            ("Avg (100)", f"{agent_stats.get('avg_score_100', 0):.1f}"),
            ("Epsilon", f"{agent_stats.get('epsilon', 1.0):.4f}"),
            ("Eps/sec", f"{self.eps_per_second:.0f}"),
            ("Memory", f"{agent_stats.get('memory_size', 0):,}"),
            ("Total Steps", f"{agent_stats.get('total_steps', 0):,}"),
        ]

        for label, value in stats:
            # Label
            label_surf = self.font_small.render(
                f"{label}:", True, (150, 150, 150)
            )
            self.screen.blit(label_surf, (x, y))

            # Value
            value_surf = self.font_medium.render(value, True, (255, 255, 255))
            self.screen.blit(value_surf, (x + 130, y - 4))

            y += 35

        # Elapsed time
        y += 10
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        time_label = self.font_small.render("Elapsed:", True, (150, 150, 150))
        time_value = self.font_medium.render(time_str, True, (100, 200, 255))
        self.screen.blit(time_label, (x, y))
        self.screen.blit(time_value, (x + 130, y - 4))

    def _draw_preview(self):
        """Draw the mini game preview."""
        if not self.preview_game_state:
            return

        x = self.preview_x
        y = self.preview_y

        # Preview border
        border_rect = pygame.Rect(x - 5, y - 30, 310, 435)
        pygame.draw.rect(
            self.screen,
            (30, 30, 45),
            border_rect,
            border_radius=10
        )

        # Title
        title = self.font_small.render("LIVE PREVIEW", True, (150, 150, 150))
        self.screen.blit(title, (x + 100, y - 25))

        # Mini game surface
        preview = self.preview_surface
        preview.fill((70, 180, 220))  # Sky

        # Scale factors
        scale_x = 300 / cfg.game_width
        scale_y = 400 / cfg.game_height

        # Draw pipes
        for pipe in self.preview_game_state.get("pipes", []):
            top_rect, bottom_rect = pipe["rects"]

            for rect in [top_rect, bottom_rect]:
                px = int(rect[0] * scale_x)
                py = int(rect[1] * scale_y)
                pw = int(rect[2] * scale_x)
                ph = int(rect[3] * scale_y)
                if ph > 0:
                    pygame.draw.rect(preview, (80, 200, 80), (px, py, pw, ph))

        # Draw bird
        bird = self.preview_game_state.get("bird", {})
        bird_rect = bird.get("rect", (0, 0, 0, 0))
        bx = int(bird_rect[0] * scale_x)
        by = int(bird_rect[1] * scale_y)
        bs = int(bird_rect[2] * scale_x)

        pygame.draw.circle(
            preview,
            (255, 220, 50),
            (bx + bs // 2, by + bs // 2),
            bs // 2
        )

        # Draw score
        score = self.preview_game_state.get("score", 0)
        score_text = self.font_large.render(str(score), True, (255, 255, 255))
        preview.blit(score_text, (150 - score_text.get_width() // 2, 20))

        # Blit preview to screen
        self.screen.blit(preview, (x, y))
        pygame.draw.rect(
            self.screen,
            (60, 60, 80),
            (x, y, 300, 400),
            width=2,
            border_radius=5
        )

    def _draw_controls_hint(self):
        """Draw keyboard controls hint."""
        hints = [
            "T: Toggle Watch Mode",
            "ESC: Return to Menu",
            "+/-: Adjust Speed",
            "S: Save Model",
            "R: Reset Training"
        ]

        y = global_config.screen_height - 25
        x = 20

        for hint in hints:
            text = self.font_small.render(hint, True, (80, 80, 100))
            self.screen.blit(text, (x, y))
            x += text.get_width() + 30
