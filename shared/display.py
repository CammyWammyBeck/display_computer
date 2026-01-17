import pygame
from .config import config


class DisplayManager:
    """Manages the pygame display and provides utility functions."""

    def __init__(self):
        self.screen = None
        self.clock = None
        self.fonts: dict[str, pygame.font.Font] = {}
        self._initialized = False

    def init(self):
        """Initialize pygame and create the display."""
        if self._initialized:
            return

        pygame.init()
        pygame.font.init()

        flags = pygame.SCALED
        if config.fullscreen:
            flags |= pygame.FULLSCREEN

        self.screen = pygame.display.set_mode(
            (config.screen_width, config.screen_height),
            flags
        )
        pygame.display.set_caption("Display Computer")

        self.clock = pygame.time.Clock()

        # Load fonts
        self._load_fonts()

        # Hide mouse cursor for clean display
        pygame.mouse.set_visible(False)

        self._initialized = True

    def _load_fonts(self):
        """Load commonly used fonts."""
        try:
            # Try to use a nice monospace font if available
            font_name = pygame.font.match_font('firacode,consolas,monaco,monospace')
            self.fonts['small'] = pygame.font.Font(font_name, 18)
            self.fonts['medium'] = pygame.font.Font(font_name, 28)
            self.fonts['large'] = pygame.font.Font(font_name, 48)
            self.fonts['title'] = pygame.font.Font(font_name, 72)
        except Exception:
            # Fall back to default font
            self.fonts['small'] = pygame.font.Font(None, 18)
            self.fonts['medium'] = pygame.font.Font(None, 28)
            self.fonts['large'] = pygame.font.Font(None, 48)
            self.fonts['title'] = pygame.font.Font(None, 72)

    def clear(self, color: tuple[int, int, int] = None):
        """Clear the screen with the background color."""
        self.screen.fill(color or config.bg_color)

    def flip(self):
        """Update the display and maintain framerate."""
        pygame.display.flip()
        self.clock.tick(config.fps)

    def get_fps(self) -> float:
        """Get the current FPS."""
        return self.clock.get_fps()

    def draw_text(
        self,
        text: str,
        x: int,
        y: int,
        font_size: str = 'medium',
        color: tuple[int, int, int] = None,
        center: bool = False
    ):
        """Draw text on the screen."""
        font = self.fonts.get(font_size, self.fonts['medium'])
        color = color or config.text_color
        surface = font.render(text, True, color)

        if center:
            rect = surface.get_rect(center=(x, y))
            self.screen.blit(surface, rect)
        else:
            self.screen.blit(surface, (x, y))

    def quit(self):
        """Clean up pygame."""
        if self._initialized:
            pygame.quit()
            self._initialized = False


# Global display manager instance
display_manager = DisplayManager()
