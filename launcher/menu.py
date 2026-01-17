import pygame
import math
from typing import Callable

from shared.config import config
from shared.display import display_manager


class MenuItem:
    """A selectable item in the menu."""

    def __init__(
        self,
        project_id: str,
        name: str,
        description: str,
        index: int
    ):
        self.project_id = project_id
        self.name = name
        self.description = description
        self.index = index

        # Animation state
        self.hover_progress = 0.0
        self.target_hover = 0.0


class Menu:
    """
    Visual menu for selecting projects.

    Displays a grid of project cards with smooth animations.
    """

    def __init__(self, on_select: Callable[[str], None]):
        self.on_select = on_select
        self.items: list[MenuItem] = []
        self.selected_index = 0
        self.visible = True

        # Layout settings
        self.card_width = 350
        self.card_height = 200
        self.card_padding = 30
        self.cards_per_row = 4

        # Animation
        self.title_offset = 0.0
        self.time = 0.0

    def set_projects(self, projects: list[dict]):
        """Set the list of available projects."""
        self.items = [
            MenuItem(
                project_id=p["id"],
                name=p["name"],
                description=p["description"],
                index=i
            )
            for i, p in enumerate(projects)
        ]
        self.selected_index = 0 if self.items else -1

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle input events."""
        if not self.visible:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                self._move_selection(1)
                return True
            elif event.key == pygame.K_LEFT:
                self._move_selection(-1)
                return True
            elif event.key == pygame.K_DOWN:
                self._move_selection(self.cards_per_row)
                return True
            elif event.key == pygame.K_UP:
                self._move_selection(-self.cards_per_row)
                return True
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self._select_current()
                return True
            elif event.key == pygame.K_ESCAPE:
                # Could be used to quit or show settings
                return True

        return False

    def _move_selection(self, delta: int):
        """Move the selection by delta positions."""
        if not self.items:
            return

        new_index = self.selected_index + delta
        if 0 <= new_index < len(self.items):
            self.selected_index = new_index

    def _select_current(self):
        """Select the currently highlighted project."""
        if 0 <= self.selected_index < len(self.items):
            item = self.items[self.selected_index]
            self.visible = False
            self.on_select(item.project_id)

    def update(self, dt: float):
        """Update animations."""
        self.time += dt

        # Update hover animations
        for item in self.items:
            item.target_hover = 1.0 if item.index == self.selected_index else 0.0
            item.hover_progress += (item.target_hover - item.hover_progress) * dt * 10

    def render(self, screen: pygame.Surface):
        """Render the menu."""
        if not self.visible:
            return

        # Draw title with subtle animation
        title_y = 80 + math.sin(self.time * 2) * 3
        display_manager.draw_text(
            "DISPLAY COMPUTER",
            config.screen_width // 2,
            int(title_y),
            font_size='title',
            color=config.primary_color,
            center=True
        )

        # Draw subtitle
        display_manager.draw_text(
            "Select a project with arrow keys, Enter to launch",
            config.screen_width // 2,
            150,
            font_size='small',
            color=(150, 150, 150),
            center=True
        )

        # Calculate grid layout
        total_width = self.cards_per_row * (self.card_width + self.card_padding) - self.card_padding
        start_x = (config.screen_width - total_width) // 2
        start_y = 220

        # Draw project cards
        for item in self.items:
            row = item.index // self.cards_per_row
            col = item.index % self.cards_per_row

            x = start_x + col * (self.card_width + self.card_padding)
            y = start_y + row * (self.card_height + self.card_padding)

            self._draw_card(screen, item, x, y)

        # Draw footer
        display_manager.draw_text(
            f"Web control: http://localhost:{config.web_port}",
            config.screen_width // 2,
            config.screen_height - 40,
            font_size='small',
            color=(100, 100, 100),
            center=True
        )

    def _draw_card(self, screen: pygame.Surface, item: MenuItem, x: int, y: int):
        """Draw a single project card."""
        # Calculate animated properties
        hover = item.hover_progress
        scale = 1.0 + hover * 0.05
        glow = int(hover * 30)

        # Adjusted dimensions for scale
        w = int(self.card_width * scale)
        h = int(self.card_height * scale)
        adj_x = x - (w - self.card_width) // 2
        adj_y = y - (h - self.card_height) // 2

        # Draw glow effect
        if glow > 0:
            glow_rect = pygame.Rect(adj_x - glow, adj_y - glow, w + glow * 2, h + glow * 2)
            glow_color = (*config.primary_color[:3], int(50 * hover))
            glow_surface = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=15)
            screen.blit(glow_surface, glow_rect.topleft)

        # Draw card background
        card_rect = pygame.Rect(adj_x, adj_y, w, h)
        bg_color = (35, 35, 50) if hover < 0.5 else (45, 45, 65)
        pygame.draw.rect(screen, bg_color, card_rect, border_radius=10)

        # Draw border
        border_color = (
            int(100 + 100 * hover),
            int(100 + 100 * hover),
            int(120 + 80 * hover)
        )
        pygame.draw.rect(screen, border_color, card_rect, width=2, border_radius=10)

        # Draw project name
        display_manager.draw_text(
            item.name,
            adj_x + w // 2,
            adj_y + 50,
            font_size='large',
            color=config.primary_color if hover > 0.5 else config.text_color,
            center=True
        )

        # Draw description (truncated)
        desc = item.description[:50] + "..." if len(item.description) > 50 else item.description
        display_manager.draw_text(
            desc,
            adj_x + w // 2,
            adj_y + 100,
            font_size='small',
            color=(180, 180, 180),
            center=True
        )

        # Draw "Press Enter" hint on selected card
        if hover > 0.5:
            display_manager.draw_text(
                "[ ENTER ]",
                adj_x + w // 2,
                adj_y + h - 35,
                font_size='small',
                color=config.success_color,
                center=True
            )

    def show(self):
        """Show the menu."""
        self.visible = True

    def hide(self):
        """Hide the menu."""
        self.visible = False
