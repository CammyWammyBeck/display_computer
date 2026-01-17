"""Renderer for Flappy Bird game."""

import math
import pygame
import numpy as np

from shared.config import config as global_config
from .config import flappy_config as cfg


class FlappyRenderer:
    """
    Renders the Flappy Bird game with RL visualizations.

    Displays the game, training stats, and optional neural network visualization.
    """

    def __init__(self, screen: pygame.Surface):
        self.screen = screen

        # Colors
        self.sky_color = (70, 180, 220)
        self.ground_color = (210, 180, 140)
        self.pipe_color = (80, 200, 80)
        self.pipe_highlight = (100, 230, 100)
        self.bird_color = (255, 220, 50)
        self.bird_eye = (0, 0, 0)

        # Layout
        self.game_offset_x = 100
        self.game_offset_y = 100
        self.game_scale = min(
            (global_config.screen_width - 500) / cfg.game_width,
            (global_config.screen_height - 200) / cfg.game_height
        )

        # Stats panel
        self.stats_x = self.game_offset_x + int(cfg.game_width * self.game_scale) + 50
        self.stats_y = 100

        # Animation
        self.time = 0.0
        self.ground_offset = 0.0

        # Fonts
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)
        self.font_score = pygame.font.Font(None, 72)

        # Pre-render static elements for performance
        self._background_surface = None
        self._init_cached_surfaces()

    def _init_cached_surfaces(self):
        """Pre-render static elements for better performance."""
        # Pre-render gradient background
        width = self._scale(cfg.game_width)
        height = self._scale(cfg.game_height)
        self._background_surface = pygame.Surface((width, height))

        for i in range(height):
            progress = i / height
            color = (
                int(70 + 30 * progress),
                int(180 - 40 * progress),
                int(220 - 80 * progress)
            )
            pygame.draw.line(
                self._background_surface,
                color,
                (0, i),
                (width, i)
            )

    def _scale(self, value: float) -> int:
        """Scale a game coordinate to screen coordinate."""
        return int(value * self.game_scale)

    def _transform(self, x: float, y: float) -> tuple[int, int]:
        """Transform game coordinates to screen coordinates."""
        return (
            self.game_offset_x + self._scale(x),
            self.game_offset_y + self._scale(y)
        )

    def render(
        self,
        game_state: dict,
        agent_stats: dict,
        show_network: bool = False,
        q_values: np.ndarray = None,
        activations: list = None,
        observation: np.ndarray = None
    ):
        """Render the complete frame."""
        self.time += 1 / 60

        # Draw game area background
        self._draw_background()

        # Draw game elements
        self._draw_pipes(game_state["pipes"])
        self._draw_bird(game_state["bird"])
        self._draw_ground()
        self._draw_score(game_state["score"])

        # Draw stats panel
        self._draw_stats(agent_stats, game_state["score"])

        # Draw neural network visualization if enabled
        if show_network and q_values is not None:
            self._draw_network_viz(q_values, activations, observation)

        # Draw game over overlay if dead
        if game_state["done"]:
            self._draw_game_over()

    def _draw_background(self):
        """Draw the sky background using pre-rendered surface."""
        # Blit pre-rendered gradient (much faster than drawing lines each frame)
        self.screen.blit(self._background_surface, (self.game_offset_x, self.game_offset_y))

        # Border
        game_rect = pygame.Rect(
            self.game_offset_x,
            self.game_offset_y,
            self._scale(cfg.game_width),
            self._scale(cfg.game_height)
        )
        pygame.draw.rect(self.screen, (50, 50, 70), game_rect, 3, border_radius=5)

    def _draw_ground(self):
        """Draw the ground with scrolling texture."""
        ground_height = 20
        ground_rect = pygame.Rect(
            self.game_offset_x,
            self.game_offset_y + self._scale(cfg.game_height) - ground_height,
            self._scale(cfg.game_width),
            ground_height
        )

        pygame.draw.rect(self.screen, self.ground_color, ground_rect)

        # Scrolling lines
        self.ground_offset = (self.ground_offset + cfg.pipe_speed) % 20
        for x in range(-20, self._scale(cfg.game_width) + 20, 20):
            line_x = self.game_offset_x + x - self.ground_offset
            pygame.draw.line(
                self.screen,
                (180, 150, 110),
                (line_x, ground_rect.top),
                (line_x + 10, ground_rect.bottom),
                2
            )

    def _draw_pipes(self, pipes: list):
        """Draw all pipes."""
        for pipe_data in pipes:
            top_rect, bottom_rect = pipe_data["rects"]

            # Top pipe
            self._draw_pipe_rect(top_rect, flip=True)

            # Bottom pipe
            self._draw_pipe_rect(bottom_rect, flip=False)

    def _draw_pipe_rect(self, rect: tuple, flip: bool):
        """Draw a single pipe with 3D effect."""
        x, y, w, h = rect
        sx, sy = self._transform(x, y)
        sw, sh = self._scale(w), self._scale(h)

        if sh <= 0:
            return

        # Main pipe body
        pipe_rect = pygame.Rect(sx, sy, sw, sh)
        pygame.draw.rect(self.screen, self.pipe_color, pipe_rect)

        # Highlight
        highlight_rect = pygame.Rect(sx + 5, sy, 10, sh)
        pygame.draw.rect(self.screen, self.pipe_highlight, highlight_rect)

        # Pipe cap
        cap_height = 25
        cap_width = sw + 10
        if flip:
            cap_y = sy + sh - cap_height
        else:
            cap_y = sy

        cap_rect = pygame.Rect(sx - 5, cap_y, cap_width, cap_height)
        pygame.draw.rect(self.screen, self.pipe_color, cap_rect)
        pygame.draw.rect(self.screen, (60, 160, 60), cap_rect, 2)

    def _draw_bird(self, bird: dict):
        """Draw the bird with animation."""
        x, y, w, h = bird["rect"]
        sx, sy = self._transform(x, y)
        sw, sh = self._scale(w), self._scale(h)

        # Rotation based on velocity
        velocity = bird["velocity"]
        angle = max(-30, min(30, velocity * 3))

        # Wing flap animation
        wing_offset = math.sin(self.time * 15) * 3

        # Body
        center = (sx + sw // 2, sy + sh // 2)

        # Create bird surface for rotation
        bird_surf = pygame.Surface((sw + 10, sh + 10), pygame.SRCALPHA)
        bird_center = (sw // 2 + 5, sh // 2 + 5)

        # Body circle
        pygame.draw.circle(bird_surf, self.bird_color, bird_center, sw // 2)

        # Wing
        wing_points = [
            (bird_center[0] - 5, bird_center[1]),
            (bird_center[0] - 15, bird_center[1] + wing_offset),
            (bird_center[0] - 5, bird_center[1] + 5),
        ]
        pygame.draw.polygon(bird_surf, (255, 200, 50), wing_points)

        # Eye
        eye_pos = (bird_center[0] + 5, bird_center[1] - 3)
        pygame.draw.circle(bird_surf, (255, 255, 255), eye_pos, 6)
        pygame.draw.circle(bird_surf, self.bird_eye, eye_pos, 3)

        # Beak
        beak_points = [
            (bird_center[0] + sw // 2, bird_center[1]),
            (bird_center[0] + sw // 2 + 10, bird_center[1] + 3),
            (bird_center[0] + sw // 2, bird_center[1] + 6),
        ]
        pygame.draw.polygon(bird_surf, (255, 150, 50), beak_points)

        # Rotate and blit
        rotated = pygame.transform.rotate(bird_surf, -angle)
        rotated_rect = rotated.get_rect(center=center)
        self.screen.blit(rotated, rotated_rect)

    def _draw_score(self, score: int):
        """Draw the current score."""
        score_text = self.font_score.render(str(score), True, (255, 255, 255))
        shadow = self.font_score.render(str(score), True, (0, 0, 0))

        x = self.game_offset_x + self._scale(cfg.game_width) // 2
        y = self.game_offset_y + 50

        # Shadow
        self.screen.blit(shadow, (x - shadow.get_width() // 2 + 2, y + 2))
        # Main text
        self.screen.blit(score_text, (x - score_text.get_width() // 2, y))

    def _draw_stats(self, agent_stats: dict, current_score: int):
        """Draw the training statistics panel."""
        x = self.stats_x
        y = self.stats_y

        # Title
        title = self.font_large.render("TRAINING STATS", True, global_config.primary_color)
        self.screen.blit(title, (x, y))
        y += 60

        # Stats
        stats = [
            ("Episode", agent_stats.get("episode", 0)),
            ("Best Score", agent_stats.get("best_score", 0)),
            ("Avg Score (100)", agent_stats.get("avg_score_100", 0)),
            ("Epsilon", f"{agent_stats.get('epsilon', 1.0):.3f}"),
            ("Memory", f"{agent_stats.get('memory_size', 0):,}"),
            ("Total Steps", f"{agent_stats.get('total_steps', 0):,}"),
        ]

        for label, value in stats:
            label_surf = self.font_medium.render(f"{label}:", True, (180, 180, 180))
            value_surf = self.font_medium.render(str(value), True, (255, 255, 255))

            self.screen.blit(label_surf, (x, y))
            self.screen.blit(value_surf, (x + 180, y))
            y += 35

        # Current score highlight
        y += 20
        pygame.draw.rect(
            self.screen,
            (40, 40, 60),
            (x - 10, y - 5, 300, 50),
            border_radius=5
        )
        score_label = self.font_medium.render("Current:", True, global_config.success_color)
        score_value = self.font_large.render(str(current_score), True, global_config.success_color)
        self.screen.blit(score_label, (x, y + 8))
        self.screen.blit(score_value, (x + 120, y))

    def _draw_network_viz(self, q_values: np.ndarray, activations: list, observation: np.ndarray = None):
        """
        Draw neural network visualization showing nodes and connections.

        Network: 4 inputs → 64 hidden → 64 hidden → 2 outputs
        """
        # Panel position and size
        panel_x = self.stats_x
        panel_y = self.stats_y + 380
        panel_width = 380
        panel_height = 320

        # Draw panel background
        pygame.draw.rect(
            self.screen,
            (25, 25, 40),
            (panel_x - 10, panel_y - 10, panel_width, panel_height),
            border_radius=10
        )

        # Title
        title = self.font_medium.render("NEURAL NETWORK", True, (150, 150, 150))
        self.screen.blit(title, (panel_x, panel_y))
        panel_y += 35

        # Network layout
        layer_sizes = [4, 8, 8, 2]  # Visualized sizes (hidden layers subsampled)
        layer_labels = ["Input", "Hidden 1", "Hidden 2", "Output"]
        layer_x_positions = [panel_x + 30, panel_x + 130, panel_x + 230, panel_x + 330]

        # Prepare activation values for each layer
        input_vals = observation if observation is not None else np.zeros(4)
        hidden1_vals = activations[0] if activations and len(activations) > 0 else np.zeros(64)
        hidden2_vals = activations[1] if activations and len(activations) > 1 else np.zeros(64)
        output_vals = q_values if q_values is not None else np.zeros(2)

        # Subsample hidden layers for visualization (show 8 representative nodes)
        hidden1_vis = hidden1_vals[::8][:8]  # Every 8th node
        hidden2_vis = hidden2_vals[::8][:8]

        layer_activations = [input_vals, hidden1_vis, hidden2_vis, output_vals]

        # Calculate node positions
        node_positions = []
        for layer_idx, (size, x_pos) in enumerate(zip(layer_sizes, layer_x_positions)):
            layer_nodes = []
            total_height = 200
            spacing = total_height / (size + 1)
            start_y = panel_y + 30

            for node_idx in range(size):
                y = start_y + spacing * (node_idx + 1)
                layer_nodes.append((x_pos, int(y)))
            node_positions.append(layer_nodes)

        # Draw connections (before nodes so they appear behind)
        for layer_idx in range(len(node_positions) - 1):
            from_layer = node_positions[layer_idx]
            to_layer = node_positions[layer_idx + 1]

            for from_pos in from_layer:
                for to_pos in to_layer:
                    # Connection color based on a simple pattern
                    alpha = 30
                    pygame.draw.line(
                        self.screen,
                        (60, 60, 80),
                        from_pos,
                        to_pos,
                        1
                    )

        # Draw nodes with activation colors
        node_radius = 12
        input_labels = ["Bird Y", "Velocity", "Pipe Dist", "Gap Dist"]
        output_labels = ["Stay", "Flap"]

        for layer_idx, (layer_nodes, acts) in enumerate(zip(node_positions, layer_activations)):
            for node_idx, (x, y) in enumerate(layer_nodes):
                # Get activation value
                if node_idx < len(acts):
                    activation = float(acts[node_idx])
                else:
                    activation = 0.0

                # Color based on activation (blue = negative, gray = zero, orange/red = positive)
                intensity = min(abs(activation) / 2.0, 1.0)  # Normalize

                if activation > 0:
                    # Positive: orange to red
                    r = int(100 + 155 * intensity)
                    g = int(100 + 100 * (1 - intensity))
                    b = int(80 * (1 - intensity))
                elif activation < 0:
                    # Negative: blue
                    r = int(80 * (1 - intensity))
                    g = int(100 + 100 * (1 - intensity))
                    b = int(100 + 155 * intensity)
                else:
                    # Zero: gray
                    r, g, b = 100, 100, 100

                node_color = (r, g, b)

                # Draw node glow for strong activations
                if intensity > 0.3:
                    glow_radius = int(node_radius + 8 * intensity)
                    glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                    glow_alpha = int(50 * intensity)
                    pygame.draw.circle(glow_surface, (*node_color, glow_alpha), (glow_radius, glow_radius), glow_radius)
                    self.screen.blit(glow_surface, (x - glow_radius, y - glow_radius))

                # Draw node
                pygame.draw.circle(self.screen, node_color, (x, y), node_radius)
                pygame.draw.circle(self.screen, (200, 200, 200), (x, y), node_radius, 1)

                # Draw labels for input/output layers
                if layer_idx == 0 and node_idx < len(input_labels):
                    label = self.font_small.render(input_labels[node_idx], True, (120, 120, 120))
                    self.screen.blit(label, (x - label.get_width() - 15, y - 8))
                elif layer_idx == 3 and node_idx < len(output_labels):
                    label = self.font_small.render(output_labels[node_idx], True, (120, 120, 120))
                    self.screen.blit(label, (x + 18, y - 8))

                    # Show Q-value next to output
                    if q_values is not None and node_idx < len(q_values):
                        q_text = self.font_small.render(f"{q_values[node_idx]:.2f}", True, node_color)
                        self.screen.blit(q_text, (x + 18, y + 8))

        # Highlight the chosen action
        if q_values is not None:
            chosen = np.argmax(q_values)
            chosen_pos = node_positions[3][chosen]
            pygame.draw.circle(self.screen, global_config.success_color, chosen_pos, node_radius + 4, 3)

    def _draw_game_over(self):
        """Draw game over overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface(
            (self._scale(cfg.game_width), self._scale(cfg.game_height)),
            pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (self.game_offset_x, self.game_offset_y))

        # Game over text
        text = self.font_large.render("GAME OVER", True, (255, 100, 100))
        x = self.game_offset_x + self._scale(cfg.game_width) // 2 - text.get_width() // 2
        y = self.game_offset_y + self._scale(cfg.game_height) // 2 - text.get_height() // 2
        self.screen.blit(text, (x, y))

        # Restarting text
        restart_text = self.font_small.render("Restarting...", True, (200, 200, 200))
        self.screen.blit(
            restart_text,
            (x + text.get_width() // 2 - restart_text.get_width() // 2, y + 50)
        )
