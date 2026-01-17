"""Flappy RL Project - Watch a neural network learn to play Flappy Bird."""

import pygame
from pathlib import Path

from projects.base_project import BaseProject, ProjectControl
from shared.config import config as global_config

from .game import FlappyGame
from .agent import DQNAgent
from .renderer import FlappyRenderer
from .config import flappy_config as cfg


class FlappyRLProject(BaseProject):
    """
    Reinforcement Learning agent learning to play Flappy Bird.

    Watch as a DQN agent learns from scratch, improving its score
    over time through trial and error.
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

        # State
        self.current_state = None
        self.show_network = False
        self.training_speed = 1
        self.paused = False
        self.restart_delay = 0

        # Setup controls
        self._setup_controls()

    def _setup_controls(self):
        """Define web UI controls."""
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
            label="Training Speed",
            value=1,
            min_value=1,
            max_value=10,
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

        # Try to load existing model
        self.agent.load("best")

        # Initial state
        self.current_state = self.game.reset()

        print("Flappy RL initialized")

    def update(self, dt: float):
        """Update game and training."""
        if self.paused:
            return

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

        # Update stats
        self._update_stats()

    def _reset_game(self):
        """Reset the game for a new episode."""
        self.current_state = self.game.reset()

    def _update_stats(self):
        """Update stats for web UI."""
        agent_stats = self.agent.get_stats()
        for key, value in agent_stats.items():
            self.stats.set(key, value)
        self.stats.set("current_score", self.game.score)

    def render(self):
        """Render the game and UI."""
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

    def _draw_pause_overlay(self):
        """Draw pause indicator."""
        font = pygame.font.Font(None, 72)
        text = font.render("PAUSED", True, (255, 255, 255))
        rect = text.get_rect(center=(
            global_config.screen_width // 2,
            global_config.screen_height // 2
        ))
        self.screen.blit(text, rect)

    def _draw_controls_hint(self):
        """Draw keyboard controls hint."""
        font = pygame.font.Font(None, 24)
        hints = [
            "ESC: Return to menu",
            "SPACE: Pause",
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

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle keyboard input."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
                return True

            elif event.key == pygame.K_n:
                self.show_network = not self.show_network
                return True

            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self.training_speed = min(10, self.training_speed + 1)
                return True

            elif event.key == pygame.K_MINUS:
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
        if command == "reset":
            self._reset_training()
            return {"status": "ok", "message": "Training reset"}

        elif command == "save":
            self.agent.save("checkpoint")
            return {"status": "ok", "message": "Model saved"}

        elif command == "load":
            success = self.agent.load("best")
            return {
                "status": "ok" if success else "error",
                "message": "Loaded best model" if success else "No model found"
            }

        elif command == "show_network":
            self.show_network = data.get("value", False)
            return {"status": "ok"}

        elif command == "speed":
            self.training_speed = int(data.get("value", 1))
            return {"status": "ok", "speed": self.training_speed}

        elif command == "pause":
            self.paused = data.get("value", False)
            return {"status": "ok", "paused": self.paused}

        return {"status": "error", "message": "Unknown command"}

    def _reset_training(self):
        """Reset the agent and start fresh."""
        self.agent = DQNAgent(save_dir=global_config.models_dir / "flappy_rl")
        self.current_state = self.game.reset()
        print("Training reset")

    def stop(self):
        """Clean up and save on exit."""
        if self.agent:
            self.agent.save("autosave")
        self.running = False
        print("Flappy RL stopped")
