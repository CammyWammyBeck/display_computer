from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pygame


@dataclass
class ProjectControl:
    """Defines a control that appears in the web UI."""

    type: str  # "button", "slider", "toggle"
    id: str
    label: str
    value: Any = None
    min_value: float = 0
    max_value: float = 100
    step: float = 1


@dataclass
class ProjectStats:
    """Stats displayed in the web UI and on-screen."""

    values: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any):
        self.values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def to_dict(self) -> dict:
        return self.values.copy()


class BaseProject(ABC):
    """
    Abstract base class that all display projects must implement.

    Projects are self-contained visualizations that can be loaded
    and unloaded by the launcher. They receive pygame events and
    web commands, and render to the provided screen.
    """

    # Override these in subclasses
    name: str = "Unnamed Project"
    description: str = "No description"
    author: str = "Unknown"
    version: str = "1.0.0"

    def __init__(self):
        self.screen: pygame.Surface = None
        self.running: bool = False
        self.stats = ProjectStats()
        self._controls: list[ProjectControl] = []

    @abstractmethod
    def setup(self, screen: pygame.Surface) -> None:
        """
        Called once when the project is loaded.
        Initialize resources, load models, etc.
        """
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Called every frame to update project state.

        Args:
            dt: Delta time in seconds since last frame
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        Called every frame to render the project.
        Draw to self.screen.
        """
        pass

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle a pygame event.

        Returns:
            True if the event was handled, False otherwise
        """
        return False

    def handle_web_command(self, command: str, data: dict) -> dict:
        """
        Handle a command from the web interface.

        Args:
            command: The command ID (e.g., "reset", "save")
            data: Additional data from the web UI

        Returns:
            Response dict to send back to web UI
        """
        return {"status": "unknown_command"}

    def get_controls(self) -> list[ProjectControl]:
        """Get the list of web UI controls for this project."""
        return self._controls

    def add_control(self, control: ProjectControl):
        """Add a control to the web UI."""
        self._controls.append(control)

    def get_stats(self) -> dict:
        """Get current stats for the web UI."""
        return self.stats.to_dict()

    def start(self, screen: pygame.Surface) -> None:
        """Start the project. Called by the launcher."""
        self.screen = screen
        self.running = True
        self.setup(screen)

    def stop(self) -> None:
        """
        Stop the project. Override to save state/models.
        Called when switching projects or shutting down.
        """
        self.running = False

    def get_info(self) -> dict:
        """Get project metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "version": self.version,
        }
