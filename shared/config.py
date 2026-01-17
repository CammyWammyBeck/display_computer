from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Global configuration for the display computer."""

    # Display settings
    screen_width: int = 1920
    screen_height: int = 1080
    fullscreen: bool = True
    fps: int = 60

    # Web server settings
    web_host: str = "0.0.0.0"
    web_port: int = 8000

    # Paths
    project_root: Path = Path(__file__).parent.parent
    models_dir: Path = project_root / "models"
    projects_dir: Path = project_root / "projects"

    # Theme colors
    bg_color: tuple[int, int, int] = (15, 15, 25)
    primary_color: tuple[int, int, int] = (100, 200, 255)
    secondary_color: tuple[int, int, int] = (255, 100, 150)
    text_color: tuple[int, int, int] = (240, 240, 240)
    success_color: tuple[int, int, int] = (100, 255, 150)
    warning_color: tuple[int, int, int] = (255, 200, 100)

    def __post_init__(self):
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)


# Global config instance
config = Config()
