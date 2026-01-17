"""Simple graph drawing utilities using pygame."""

from dataclasses import dataclass, field
from collections import deque
import pygame


@dataclass
class GraphStyle:
    """Styling options for graphs."""

    bg_color: tuple[int, int, int] = (25, 25, 40)
    border_color: tuple[int, int, int] = (60, 60, 80)
    grid_color: tuple[int, int, int] = (40, 40, 55)
    line_color: tuple[int, int, int] = (100, 200, 255)
    fill_color: tuple[int, int, int, int] = (100, 200, 255, 30)
    text_color: tuple[int, int, int] = (150, 150, 150)
    title_color: tuple[int, int, int] = (200, 200, 200)
    line_width: int = 2
    show_grid: bool = True
    show_fill: bool = True
    grid_lines: int = 4


class LineGraph:
    """
    A simple line graph renderer for pygame.

    Maintains a history buffer and renders to a specified rect.
    """

    def __init__(
        self,
        max_points: int = 500,
        title: str = "",
        style: GraphStyle = None,
        y_min: float = None,
        y_max: float = None,
    ):
        self.max_points = max_points
        self.title = title
        self.style = style or GraphStyle()
        self.y_min_fixed = y_min
        self.y_max_fixed = y_max

        self.data: deque[float] = deque(maxlen=max_points)
        self.font = None
        self.title_font = None

    def _ensure_fonts(self):
        """Lazily initialize fonts."""
        if self.font is None:
            self.font = pygame.font.Font(None, 20)
            self.title_font = pygame.font.Font(None, 24)

    def add_point(self, value: float):
        """Add a data point to the graph."""
        self.data.append(value)

    def add_points(self, values: list[float]):
        """Add multiple data points."""
        for v in values:
            self.data.append(v)

    def clear(self):
        """Clear all data."""
        self.data.clear()

    def render(self, screen: pygame.Surface, rect: pygame.Rect):
        """Render the graph to the screen."""
        self._ensure_fonts()

        # Draw background
        pygame.draw.rect(screen, self.style.bg_color, rect, border_radius=8)
        pygame.draw.rect(screen, self.style.border_color, rect, width=1, border_radius=8)

        if len(self.data) < 2:
            # Not enough data
            self._draw_title(screen, rect)
            self._draw_no_data(screen, rect)
            return

        # Calculate bounds
        padding = 40
        graph_rect = pygame.Rect(
            rect.x + padding,
            rect.y + 30,
            rect.width - padding - 10,
            rect.height - 50
        )

        # Calculate y range
        data_list = list(self.data)
        y_min = self.y_min_fixed if self.y_min_fixed is not None else min(data_list)
        y_max = self.y_max_fixed if self.y_max_fixed is not None else max(data_list)

        # Avoid division by zero
        if y_max == y_min:
            y_max = y_min + 1

        # Draw grid
        if self.style.show_grid:
            self._draw_grid(screen, graph_rect, y_min, y_max)

        # Draw line
        self._draw_line(screen, graph_rect, data_list, y_min, y_max)

        # Draw title
        self._draw_title(screen, rect)

        # Draw current value
        self._draw_current_value(screen, rect, data_list[-1])

    def _draw_grid(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        y_min: float,
        y_max: float
    ):
        """Draw grid lines and labels."""
        # Horizontal grid lines
        for i in range(self.style.grid_lines + 1):
            y = rect.y + (rect.height * i // self.style.grid_lines)
            pygame.draw.line(
                screen,
                self.style.grid_color,
                (rect.x, y),
                (rect.x + rect.width, y),
                1
            )

            # Y-axis label
            value = y_max - (y_max - y_min) * i / self.style.grid_lines
            label = self._format_value(value)
            text = self.font.render(label, True, self.style.text_color)
            screen.blit(text, (rect.x - 35, y - 8))

    def _draw_line(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        data: list[float],
        y_min: float,
        y_max: float
    ):
        """Draw the data line and optional fill."""
        if len(data) < 2:
            return

        # Calculate points
        points = []
        for i, value in enumerate(data):
            x = rect.x + (i * rect.width // (len(data) - 1))
            y = rect.y + rect.height - int(
                (value - y_min) / (y_max - y_min) * rect.height
            )
            y = max(rect.y, min(rect.y + rect.height, y))
            points.append((x, y))

        # Draw fill
        if self.style.show_fill and len(points) >= 2:
            fill_points = (
                [(points[0][0], rect.y + rect.height)] +
                points +
                [(points[-1][0], rect.y + rect.height)]
            )
            fill_surface = pygame.Surface(
                (rect.width + 1, rect.height + 1),
                pygame.SRCALPHA
            )
            adjusted_points = [
                (p[0] - rect.x, p[1] - rect.y) for p in fill_points
            ]
            pygame.draw.polygon(fill_surface, self.style.fill_color, adjusted_points)
            screen.blit(fill_surface, (rect.x, rect.y))

        # Draw line
        if len(points) >= 2:
            pygame.draw.lines(
                screen,
                self.style.line_color,
                False,
                points,
                self.style.line_width
            )

    def _draw_title(self, screen: pygame.Surface, rect: pygame.Rect):
        """Draw the graph title."""
        if self.title:
            text = self.title_font.render(self.title, True, self.style.title_color)
            screen.blit(text, (rect.x + 10, rect.y + 6))

    def _draw_current_value(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        value: float
    ):
        """Draw the current value in the corner."""
        label = self._format_value(value)
        text = self.font.render(label, True, self.style.line_color)
        screen.blit(text, (rect.x + rect.width - text.get_width() - 10, rect.y + 8))

    def _draw_no_data(self, screen: pygame.Surface, rect: pygame.Rect):
        """Draw a 'no data' message."""
        text = self.font.render("Waiting for data...", True, self.style.text_color)
        x = rect.x + (rect.width - text.get_width()) // 2
        y = rect.y + (rect.height - text.get_height()) // 2
        screen.blit(text, (x, y))

    def _format_value(self, value: float) -> str:
        """Format a value for display."""
        if abs(value) >= 1000:
            return f"{value / 1000:.1f}k"
        elif abs(value) >= 1:
            return f"{value:.1f}"
        else:
            return f"{value:.3f}"


class MultiLineGraph(LineGraph):
    """Graph that can display multiple data series."""

    def __init__(
        self,
        max_points: int = 500,
        title: str = "",
        style: GraphStyle = None,
        y_min: float = None,
        y_max: float = None,
        series_colors: list[tuple[int, int, int]] = None,
        series_names: list[str] = None,
    ):
        super().__init__(max_points, title, style, y_min, y_max)

        self.series_colors = series_colors or [
            (100, 200, 255),
            (255, 100, 150),
            (100, 255, 150),
            (255, 200, 100),
        ]
        self.series_names = series_names or []
        self.series_data: list[deque[float]] = []

    def add_series(self, name: str = None) -> int:
        """Add a new data series. Returns the series index."""
        self.series_data.append(deque(maxlen=self.max_points))
        if name:
            self.series_names.append(name)
        return len(self.series_data) - 1

    def add_point_to_series(self, series_index: int, value: float):
        """Add a point to a specific series."""
        if 0 <= series_index < len(self.series_data):
            self.series_data[series_index].append(value)

    def render(self, screen: pygame.Surface, rect: pygame.Rect):
        """Render all series."""
        self._ensure_fonts()

        # Draw background
        pygame.draw.rect(screen, self.style.bg_color, rect, border_radius=8)
        pygame.draw.rect(screen, self.style.border_color, rect, width=1, border_radius=8)

        # Check if we have any data
        has_data = any(len(s) >= 2 for s in self.series_data)
        if not has_data:
            self._draw_title(screen, rect)
            self._draw_no_data(screen, rect)
            return

        # Calculate bounds
        padding = 40
        graph_rect = pygame.Rect(
            rect.x + padding,
            rect.y + 30,
            rect.width - padding - 10,
            rect.height - 50
        )

        # Calculate y range across all series
        all_values = [v for s in self.series_data for v in s]
        y_min = self.y_min_fixed if self.y_min_fixed is not None else min(all_values)
        y_max = self.y_max_fixed if self.y_max_fixed is not None else max(all_values)

        if y_max == y_min:
            y_max = y_min + 1

        # Draw grid
        if self.style.show_grid:
            self._draw_grid(screen, graph_rect, y_min, y_max)

        # Draw each series
        for i, series in enumerate(self.series_data):
            if len(series) >= 2:
                color = self.series_colors[i % len(self.series_colors)]
                original_color = self.style.line_color
                self.style.line_color = color
                self.style.show_fill = False
                self._draw_line(screen, graph_rect, list(series), y_min, y_max)
                self.style.line_color = original_color
                self.style.show_fill = True

        # Draw title
        self._draw_title(screen, rect)

        # Draw legend
        self._draw_legend(screen, rect)

    def _draw_legend(self, screen: pygame.Surface, rect: pygame.Rect):
        """Draw the legend."""
        x = rect.x + rect.width - 10
        y = rect.y + 8

        for i, name in enumerate(self.series_names):
            if i >= len(self.series_data):
                break

            color = self.series_colors[i % len(self.series_colors)]
            text = self.font.render(name, True, color)
            x -= text.get_width() + 20
            screen.blit(text, (x + 15, y))
            pygame.draw.rect(screen, color, (x, y + 3, 10, 10))
