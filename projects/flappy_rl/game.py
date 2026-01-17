"""Flappy Bird game environment."""

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import flappy_config as cfg


@dataclass
class Bird:
    """The flappy bird."""

    y: float = 0.0
    velocity: float = 0.0
    alive: bool = True

    def __post_init__(self):
        self.y = cfg.game_height / 2

    def flap(self):
        """Make the bird flap."""
        if self.alive:
            self.velocity = cfg.flap_strength

    def update(self):
        """Update bird physics."""
        if not self.alive:
            return

        self.velocity += cfg.gravity
        self.y += self.velocity

        # Check bounds
        if self.y < 0 or self.y > cfg.game_height - cfg.bird_size:
            self.alive = False

    def get_rect(self) -> tuple[float, float, float, float]:
        """Get bird hitbox (x, y, width, height)."""
        return (cfg.bird_x, self.y, cfg.bird_size, cfg.bird_size)


@dataclass
class Pipe:
    """A pair of pipes (top and bottom)."""

    x: float
    gap_y: float  # Center of the gap

    def update(self):
        """Move pipe to the left."""
        self.x -= cfg.pipe_speed

    def is_offscreen(self) -> bool:
        """Check if pipe has moved off screen."""
        return self.x < -cfg.pipe_width

    def get_rects(self) -> tuple[tuple, tuple]:
        """Get top and bottom pipe hitboxes."""
        gap_top = self.gap_y - cfg.pipe_gap / 2
        gap_bottom = self.gap_y + cfg.pipe_gap / 2

        top_pipe = (self.x, 0, cfg.pipe_width, gap_top)
        bottom_pipe = (self.x, gap_bottom, cfg.pipe_width, cfg.game_height - gap_bottom)

        return top_pipe, bottom_pipe


class FlappyGame:
    """
    Flappy Bird game environment.

    Can be used standalone or as a gym-like environment for RL.
    """

    def __init__(self):
        self.bird: Bird = None
        self.pipes: list[Pipe] = []
        self.score: int = 0
        self.frame: int = 0
        self.done: bool = False

        # Track which pipes have been passed for scoring
        self._passed_pipes: set[int] = set()

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game and return initial observation."""
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.frame = 0
        self.done = False
        self._passed_pipes = set()

        # Spawn initial pipe
        self._spawn_pipe()

        return self._get_observation()

    def _spawn_pipe(self):
        """Spawn a new pipe pair."""
        # Random gap position, avoiding edges
        margin = 100
        gap_y = random.randint(
            margin + cfg.pipe_gap // 2,
            cfg.game_height - margin - cfg.pipe_gap // 2
        )
        pipe = Pipe(x=cfg.game_width + 50, gap_y=gap_y)
        self.pipes.append(pipe)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: 0 = do nothing, 1 = flap

        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self._get_observation(), 0.0, True, {}

        # Apply action
        if action == 1:
            self.bird.flap()

        # Update game state
        self.bird.update()

        reward = cfg.reward_alive

        # Update pipes
        for pipe in self.pipes:
            pipe.update()

            # Check for scoring
            pipe_id = id(pipe)
            if pipe.x + cfg.pipe_width < cfg.bird_x and pipe_id not in self._passed_pipes:
                self.score += 1
                self._passed_pipes.add(pipe_id)
                reward += cfg.reward_pipe

        # Remove offscreen pipes
        self.pipes = [p for p in self.pipes if not p.is_offscreen()]

        # Spawn new pipes
        self.frame += 1
        if self.frame % cfg.pipe_spawn_interval == 0:
            self._spawn_pipe()

        # Check collisions
        if self._check_collision():
            self.bird.alive = False
            self.done = True
            reward = cfg.reward_death

        # Check if bird died from bounds
        if not self.bird.alive:
            self.done = True
            reward = cfg.reward_death

        return self._get_observation(), reward, self.done, {"score": self.score}

    def _check_collision(self) -> bool:
        """Check if bird collides with any pipe."""
        bird_rect = self.bird.get_rect()

        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.get_rects()

            if self._rects_collide(bird_rect, top_rect):
                return True
            if self._rects_collide(bird_rect, bottom_rect):
                return True

        return False

    def _rects_collide(
        self,
        r1: tuple[float, float, float, float],
        r2: tuple[float, float, float, float]
    ) -> bool:
        """Check if two rectangles collide."""
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2

        return (
            x1 < x2 + w2 and
            x1 + w1 > x2 and
            y1 < y2 + h2 and
            y1 + h1 > y2
        )

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation for the RL agent.

        Returns normalized values:
        - Bird y position (0-1)
        - Bird velocity (normalized)
        - Horizontal distance to next pipe (0-1)
        - Vertical distance to gap center (normalized)
        """
        # Find next pipe (the one the bird hasn't passed yet)
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + cfg.pipe_width > cfg.bird_x:
                next_pipe = pipe
                break

        if next_pipe is None:
            # No pipe ahead, use defaults
            pipe_dist = 1.0
            gap_dist = 0.0
        else:
            pipe_dist = (next_pipe.x - cfg.bird_x) / cfg.game_width
            gap_dist = (next_pipe.gap_y - self.bird.y) / cfg.game_height

        obs = np.array([
            self.bird.y / cfg.game_height,
            self.bird.velocity / 15.0,  # Normalize velocity
            pipe_dist,
            gap_dist,
        ], dtype=np.float32)

        return obs

    def get_state(self) -> dict:
        """Get full game state for rendering."""
        return {
            "bird": {
                "y": self.bird.y,
                "velocity": self.bird.velocity,
                "alive": self.bird.alive,
                "rect": self.bird.get_rect(),
            },
            "pipes": [
                {
                    "x": p.x,
                    "gap_y": p.gap_y,
                    "rects": p.get_rects(),
                }
                for p in self.pipes
            ],
            "score": self.score,
            "done": self.done,
        }
