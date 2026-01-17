from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum, auto
import asyncio
from collections import defaultdict


class EventType(Enum):
    """Types of events that can be dispatched."""

    # Project lifecycle
    PROJECT_START = auto()
    PROJECT_STOP = auto()
    PROJECT_STATS_UPDATE = auto()

    # Web control
    WEB_COMMAND = auto()
    WEB_CONNECTED = auto()
    WEB_DISCONNECTED = auto()

    # Launcher
    MENU_OPEN = auto()
    MENU_SELECT = auto()

    # System
    SHUTDOWN = auto()
    CONFIG_CHANGE = auto()


@dataclass
class Event:
    """An event that can be dispatched through the event bus."""

    type: EventType
    data: dict = field(default_factory=dict)
    source: str = "system"


class EventBus:
    """
    Simple event bus for communication between components.
    Supports both sync and async handlers.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handlers = defaultdict(list)
            cls._instance._async_handlers = defaultdict(list)
        return cls._instance

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """Subscribe a sync handler to an event type."""
        self._handlers[event_type].append(handler)

    def subscribe_async(self, event_type: EventType, handler: Callable[[Event], Any]):
        """Subscribe an async handler to an event type."""
        self._async_handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Remove a handler from an event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
        if handler in self._async_handlers[event_type]:
            self._async_handlers[event_type].remove(handler)

    def emit(self, event: Event):
        """Emit an event to all subscribed sync handlers."""
        for handler in self._handlers[event.type]:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in event handler: {e}")

    async def emit_async(self, event: Event):
        """Emit an event to all subscribed handlers (sync and async)."""
        # Call sync handlers
        self.emit(event)

        # Call async handlers
        for handler in self._async_handlers[event.type]:
            try:
                await handler(event)
            except Exception as e:
                print(f"Error in async event handler: {e}")

    def clear(self):
        """Clear all handlers."""
        self._handlers.clear()
        self._async_handlers.clear()


# Global event bus instance
event_bus = EventBus()
