"""Configuration for Flappy Bird RL project."""

from dataclasses import dataclass


@dataclass
class FlappyConfig:
    # Game physics
    gravity: float = 0.5
    flap_strength: float = -8.0
    bird_x: int = 150
    bird_size: int = 30

    # Pipes
    pipe_width: int = 80
    pipe_gap: int = 180
    pipe_speed: float = 4.0
    pipe_spawn_interval: int = 90  # frames

    # Game area
    game_width: int = 600
    game_height: int = 800

    # RL hyperparameters
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    memory_size: int = 100000
    target_update: int = 10  # Episodes between target network updates

    # Training
    train_speed_multiplier: int = 1  # How many game steps per frame
    num_parallel_birds: int = 1  # For visualization, show one bird

    # Training mode (fast training with graphs instead of game render)
    training_mode: bool = False
    episodes_per_frame: int = 10  # Full episodes to run per render frame in training mode
    max_episodes_per_frame: int = 50  # Maximum for speed slider
    graph_history_length: int = 500  # Episodes to show in graphs
    preview_interval: int = 50  # Show mini game preview every N episodes
    preview_duration: int = 1  # How many episodes to show in preview

    # Rewards
    reward_alive: float = 0.1
    reward_pipe: float = 10.0
    reward_death: float = -10.0

    # Visualization
    show_hitboxes: bool = False
    show_network: bool = False


flappy_config = FlappyConfig()
