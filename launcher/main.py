import pygame
import asyncio
from typing import Optional

from shared.config import config
from shared.display import display_manager
from shared.events import event_bus, Event, EventType
from projects.base_project import BaseProject
from .project_loader import ProjectLoader
from .menu import Menu


class Launcher:
    """
    Main launcher application.

    Manages the display, menu, and project lifecycle.
    Runs the main game loop and coordinates between components.
    """

    def __init__(self):
        self.project_loader = ProjectLoader()
        self.menu = Menu(on_select=self._on_project_selected)
        self.current_project: Optional[BaseProject] = None
        self.running = False

        # Subscribe to events
        event_bus.subscribe(EventType.WEB_COMMAND, self._handle_web_command)
        event_bus.subscribe(EventType.SHUTDOWN, self._handle_shutdown)

    def _on_project_selected(self, project_id: str):
        """Called when a project is selected from the menu."""
        self._load_project(project_id)

    def _load_project(self, project_id: str):
        """Load and start a project."""
        # Stop current project if running
        if self.current_project:
            self.current_project.stop()
            self.current_project = None

        # Load new project
        project = self.project_loader.get_project(project_id)
        if project:
            print(f"Starting project: {project.name}")
            project.start(display_manager.screen)
            self.current_project = project
            self.menu.hide()

            event_bus.emit(Event(
                type=EventType.PROJECT_START,
                data={"project_id": project_id, "name": project.name}
            ))
        else:
            print(f"Failed to load project: {project_id}")
            self.menu.show()

    def _return_to_menu(self):
        """Return to the main menu."""
        if self.current_project:
            self.current_project.stop()
            self.current_project = None

            event_bus.emit(Event(
                type=EventType.PROJECT_STOP,
                data={}
            ))

        self.menu.show()

    def _handle_web_command(self, event: Event):
        """Handle commands from the web interface."""
        command = event.data.get("command")
        data = event.data.get("data", {})

        if command == "select_project":
            project_id = data.get("project_id")
            if project_id:
                self._load_project(project_id)

        elif command == "return_to_menu":
            self._return_to_menu()

        elif command == "shutdown":
            self.running = False

        elif self.current_project:
            # Forward command to current project
            response = self.current_project.handle_web_command(command, data)
            return response

    def _handle_shutdown(self, event: Event):
        """Handle shutdown event."""
        self.running = False

    def init(self):
        """Initialize the launcher."""
        display_manager.init()

        # Discover available projects
        projects = self.project_loader.discover_projects()
        self.menu.set_projects(projects)

        print(f"Discovered {len(projects)} projects")

    def run(self):
        """Run the main loop."""
        self.running = True
        last_time = pygame.time.get_ticks()

        while self.running:
            # Calculate delta time
            current_time = pygame.time.get_ticks()
            dt = (current_time - last_time) / 1000.0
            last_time = current_time

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break

                # Global escape to return to menu
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.current_project:
                            self._return_to_menu()
                            continue
                        else:
                            # Escape in menu could quit
                            pass

                    # Debug: Q to quit
                    if event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        self.running = False
                        break

                # Let menu or project handle the event
                if self.menu.visible:
                    self.menu.handle_event(event)
                elif self.current_project:
                    self.current_project.handle_event(event)

            # Update
            if self.menu.visible:
                self.menu.update(dt)
            elif self.current_project:
                self.current_project.update(dt)

                # Emit stats update for web UI
                event_bus.emit(Event(
                    type=EventType.PROJECT_STATS_UPDATE,
                    data={"stats": self.current_project.get_stats()}
                ))

            # Render
            display_manager.clear()

            if self.menu.visible:
                self.menu.render(display_manager.screen)
            elif self.current_project:
                self.current_project.render()

            display_manager.flip()

        # Cleanup
        if self.current_project:
            self.current_project.stop()
        display_manager.quit()

    def get_state(self) -> dict:
        """Get current launcher state for web UI."""
        return {
            "menu_visible": self.menu.visible,
            "current_project": self.current_project.get_info() if self.current_project else None,
            "available_projects": [
                {
                    "id": pid,
                    "name": getattr(self.project_loader.projects[pid], 'name', pid),
                    "description": getattr(self.project_loader.projects[pid], 'description', ''),
                }
                for pid in self.project_loader.list_projects()
            ]
        }
