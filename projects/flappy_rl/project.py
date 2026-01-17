"""Flappy RL Project - Watch a neural network learn to play Flappy Bird."""

import numpy as np
import pygame
from pathlib import Path

from projects.base_project import BaseProject, ProjectControl
from shared.config import config as global_config

from .game import FlappyGame
from .agent import DQNAgent
from .renderer import FlappyRenderer
from .training_renderer import TrainingRenderer
from .config import flappy_config as cfg


class FlappyRLProject(BaseProject):
    """
    Reinforcement Learning agent learning to play Flappy Bird.

    Watch as a DQN agent learns from scratch, improving its score
    over time through trial and error.

    Modes:
    - Watch Mode: See the agent play in real-time (slower training)
    - Training Mode: Fast training with graphs and statistics
    """

    name = "Flappy RL"
    description = "Watch AI learn Flappy Bird in real-time"
    author = "Display Computer"
    version = "1.0.0"

    def __init__(self):
        super().__init__()

        self.game: FlappyGame = None
        self.agent: DQNAgent = None
        self.renderer: FlappyRenderer = None
        self.training_renderer: TrainingRenderer = None

        # State
        self.current_state = None
        self.show_network = False
        self.training_speed = 1
        self.paused = False
        self.restart_delay = 0

        # Training mode
        self.training_mode = False
        self.episodes_per_frame = cfg.episodes_per_frame
        self.preview_counter = 0
        self.last_episode = 0

        # Setup controls
        self._setup_controls()

    def _setup_controls(self):
        """Define web UI controls."""
        self.add_control(ProjectControl(
            type="toggle",
            id="training_mode",
            label="Training Mode",
            value=False
        ))
        self.add_control(ProjectControl(
            type="button",
            id="reset",
            label="Reset Training"
        ))
        self.add_control(ProjectControl(
            type="button",
            id="save",
            label="Save Model"
        ))
        self.add_control(ProjectControl(
            type="button",
            id="load",
            label="Load Best"
        ))
        self.add_control(ProjectControl(
            type="toggle",
            id="show_network",
            label="Show Network",
            value=False
        ))
        self.add_control(ProjectControl(
            type="slider",
            id="speed",
            label="Speed",
            value=1,
            min_value=1,
            max_value=cfg.max_episodes_per_frame,
            step=1
        ))
        self.add_control(ProjectControl(
            type="toggle",
            id="pause",
            label="Pause",
            value=False
        ))

    def setup(self, screen: pygame.Surface):
        """Initialize the project."""
        self.screen = screen

        # Initialize components
        self.game = FlappyGame()
        self.agent = DQNAgent(save_dir=global_config.models_dir / "flappy_rl")
        self.renderer = FlappyRenderer(screen)
        self.training_renderer = TrainingRenderer(screen)

        # Try to load existing model and restore history
        if self.agent.load("best"):
            self._restore_graph_history()

        # Initial state
        self.current_state = self.game.reset()

        print("Flappy RL initialized")

    def _restore_graph_history(self):
        """Restore training history to graphs after loading a model."""
        history = self.agent.get_history()
        self.training_renderer.load_history(history)

    def update(self, dt: float):
        """Update game and training."""
        if self.paused:
            return

        if self.training_mode:
            self._update_training_mode()
        else:
            self._update_watch_mode()

        # Update stats
        self._update_stats()

    def _update_watch_mode(self):
        """Update in watch mode - render every frame, slower training."""
        # Handle restart delay after death
        if self.restart_delay > 0:
            self.restart_delay -= 1
            if self.restart_delay == 0:
                self._reset_game()
            return

        # Run multiple steps per frame for faster training
        for _ in range(self.training_speed):
            if self.game.done:
                break

            # Agent selects action
            action = self.agent.select_action(self.current_state)

            # Take step in environment
            next_state, reward, done, info = self.game.step(action)

            # Store transition and train
            self.agent.store_transition(
                self.current_state,
                action,
                reward,
                next_state,
                done
            )
            self.agent.train_step()

            self.current_state = next_state

            if done:
                self.agent.end_episode(info["score"])
                self.restart_delay = 60  # Show death for 1 second
                break

    def _update_training_mode(self):
        """Update in training mode - run many episodes per frame."""
        episodes_completed = 0
        steps_this_frame = 0
        max_steps = 10000  # Safety limit per frame

        while episodes_completed < self.episodes_per_frame and steps_this_frame < max_steps:
            # Agent selects action
            action = self.agent.select_action(self.current_state)

            # Take step in environment
            next_state, reward, done, info = self.game.step(action)

            # Store transition and train
            self.agent.store_transition(
                self.current_state,
                action,
                reward,
                next_state,
                done
            )
            self.agent.train_step()

            self.current_state = next_state
            steps_this_frame += 1

            if done:
                # Episode finished
                score = info["score"]
                self.agent.end_episode(score)
                episodes_completed += 1

                # Add data to training renderer
                stats = self.agent.get_stats()
                self.training_renderer.add_episode_data(
                    score=score,
                    avg_score=stats.get("avg_score_100", 0),
                    epsilon=stats.get("epsilon", 1.0),
                    loss=stats.get("avg_loss", 0)
                )

                # Check if we should show preview
                self.preview_counter += 1
                if self.preview_counter >= cfg.preview_interval:
                    self.preview_counter = 0
                    # Show preview for next episode
                    self._show_preview_episode()

                # Reset for next episode
                self.current_state = self.game.reset()

    def _show_preview_episode(self):
        """Run one episode with preview updates."""
        # Just set the current game state for preview
        self.training_renderer.set_preview_state(self.game.get_state())

    def _reset_game(self):
        """Reset the game for a new episode."""
        self.current_state = self.game.reset()

    def _update_stats(self):
        """Update stats for web UI."""
        agent_stats = self.agent.get_stats()
        for key, value in agent_stats.items():
            self.stats.set(key, value)
        self.stats.set("current_score", self.game.score)
        self.stats.set("training_mode", self.training_mode)

    def render(self):
        """Render the game and UI."""
        if self.training_mode:
            self._render_training_mode()
        else:
            self._render_watch_mode()

    def _render_watch_mode(self):
        """Render in watch mode - show the game."""
        # Get Q-values for visualization
        q_values = None
        activations = None
        if self.show_network and self.current_state is not None:
            q_values = self.agent.get_q_values(self.current_state)
            activations = self.agent.get_network_activations(self.current_state)

        # Render
        self.renderer.render(
            game_state=self.game.get_state(),
            agent_stats=self.agent.get_stats(),
            show_network=self.show_network,
            q_values=q_values,
            activations=activations
        )

        # Draw pause overlay
        if self.paused:
            self._draw_pause_overlay()

        # Draw controls hint
        self._draw_controls_hint()

    def _render_training_mode(self):
        """Render in training mode - show graphs and stats."""
        # Update preview if running
        if self.training_renderer.show_preview:
            self.training_renderer.set_preview_state(self.game.get_state())

        # Render training visualization
        self.training_renderer.render(
            agent_stats=self.agent.get_stats(),
            current_score=self.game.score
        )

        # Draw pause overlay
        if self.paused:
            self._draw_pause_overlay()

    def _draw_pause_overlay(self):
        """Draw pause indicator."""
        font = pygame.font.Font(None, 72)
        text = font.render("PAUSED", True, (255, 255, 255))
        rect = text.get_rect(center=(
            global_config.screen_width // 2,
            global_config.screen_height // 2
        ))

        # Semi-transparent background
        overlay = pygame.Surface(
            (global_config.screen_width, global_config.screen_height),
            pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, rect)

    def _draw_controls_hint(self):
        """Draw keyboard controls hint."""
        font = pygame.font.Font(None, 24)
        hints = [
            "ESC: Return to menu",
            "SPACE: Pause",
            "T: Training mode",
            "N: Toggle network view",
            "+/-: Adjust speed",
            "R: Reset training",
            "S: Save model"
        ]

        y = global_config.screen_height - 30
        for hint in hints:
            text = font.render(hint, True, (100, 100, 100))
            self.screen.blit(text, (20, y))
            y -= 25

    def _toggle_training_mode(self):
        """Toggle between watch and training mode."""
        self.training_mode = not self.training_mode

        if self.training_mode:
            # Entering training mode
            print("Switched to Training Mode")
            self.training_renderer.hide_preview()
        else:
            # Entering watch mode
            print("Switched to Watch Mode")
            # Reset game to show current agent performance
            self.current_state = self.game.reset()
            self.restart_delay = 0

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle keyboard input."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
                return True

            elif event.key == pygame.K_t:
                self._toggle_training_mode()
                return True

            elif event.key == pygame.K_n:
                self.show_network = not self.show_network
                return True

            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                if self.training_mode:
                    self.episodes_per_frame = min(
                        cfg.max_episodes_per_frame,
                        self.episodes_per_frame + 5
                    )
                    cfg.episodes_per_frame = self.episodes_per_frame
                else:
                    self.training_speed = min(10, self.training_speed + 1)
                return True

            elif event.key == pygame.K_MINUS:
                if self.training_mode:
                    self.episodes_per_frame = max(1, self.episodes_per_frame - 5)
                    cfg.episodes_per_frame = self.episodes_per_frame
                else:
                    self.training_speed = max(1, self.training_speed - 1)
                return True

            elif event.key == pygame.K_r:
                self._reset_training()
                return True

            elif event.key == pygame.K_s:
                self.agent.save("checkpoint")
                return True

        return False

    def handle_web_command(self, command: str, data: dict) -> dict:
        """Handle commands from web UI."""
        if command == "training_mode":
            self.training_mode = data.get("value", False)
            if not self.training_mode:
                self.current_state = self.game.reset()
            return {"status": "ok", "training_mode": self.training_mode}

        elif command == "reset":
            self._reset_training()
            return {"status": "ok", "message": "Training reset"}

        elif command == "save":
            self.agent.save("checkpoint")
            return {"status": "ok", "message": "Model saved"}

        elif command == "load":
            success = self.agent.load("best")
            if success:
                self._restore_graph_history()
            return {
                "status": "ok" if success else "error",
                "message": "Loaded best model" if success else "No model found"
            }

        elif command == "show_network":
            self.show_network = data.get("value", False)
            return {"status": "ok"}

        elif command == "speed":
            value = int(data.get("value", 1))
            if self.training_mode:
                self.episodes_per_frame = value
                cfg.episodes_per_frame = value
            else:
                self.training_speed = value
            return {"status": "ok", "speed": value}

        elif command == "pause":
            self.paused = data.get("value", False)
            return {"status": "ok", "paused": self.paused}

        return {"status": "error", "message": "Unknown command"}

    def _reset_training(self):
        """Reset the agent and start fresh."""
        self.agent = DQNAgent(save_dir=global_config.models_dir / "flappy_rl")
        self.current_state = self.game.reset()

        # Clear training renderer graphs
        self.training_renderer.score_graph.clear()
        self.training_renderer.avg_graph.clear()
        self.training_renderer.epsilon_graph.clear()
        self.training_renderer.loss_graph.clear()

        print("Training reset")

    def stop(self):
        """Clean up and save on exit."""
        if self.agent:
            self.agent.save("autosave")
        self.running = False
        print("Flappy RL stopped")
