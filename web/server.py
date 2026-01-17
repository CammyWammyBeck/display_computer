"""FastAPI web server for remote control."""

import asyncio
import json
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from shared.config import config
from shared.events import event_bus, Event, EventType


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        event_bus.emit(Event(type=EventType.WEB_CONNECTED))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            event_bus.emit(Event(type=EventType.WEB_DISCONNECTED))

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


class WebServer:
    """
    FastAPI server for web-based control.

    Provides:
    - REST API for project info and control
    - WebSocket for real-time stats and commands
    - Static file serving for the frontend
    """

    def __init__(self, get_state: Callable[[], dict]):
        self.get_state = get_state
        self.app = FastAPI(title="Display Computer")
        self.manager = ConnectionManager()

        self._setup_routes()
        self._setup_event_handlers()

        # Stats broadcast task
        self._broadcast_task = None

    def _setup_routes(self):
        """Setup API routes."""
        static_dir = Path(__file__).parent / "static"

        # Serve static files
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve the main page."""
            index_path = static_dir / "index.html"
            return FileResponse(index_path)

        @self.app.get("/api/state")
        async def get_state():
            """Get current launcher state."""
            return self.get_state()

        @self.app.post("/api/command/{command}")
        async def send_command(command: str, data: dict = None):
            """Send a command to the launcher."""
            event_bus.emit(Event(
                type=EventType.WEB_COMMAND,
                data={"command": command, "data": data or {}},
                source="web"
            ))
            return {"status": "ok"}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication."""
            await self.manager.connect(websocket)

            # Send initial state
            await websocket.send_json({
                "type": "state",
                "data": self.get_state()
            })

            try:
                while True:
                    # Receive commands from client
                    data = await websocket.receive_json()
                    await self._handle_ws_message(data)

            except WebSocketDisconnect:
                self.manager.disconnect(websocket)
            except Exception as e:
                print(f"WebSocket error: {e}")
                self.manager.disconnect(websocket)

    async def _handle_ws_message(self, message: dict):
        """Handle incoming WebSocket message."""
        msg_type = message.get("type")

        if msg_type == "command":
            command = message.get("command")
            data = message.get("data", {})

            event_bus.emit(Event(
                type=EventType.WEB_COMMAND,
                data={"command": command, "data": data},
                source="web"
            ))

        elif msg_type == "ping":
            # Keepalive
            pass

    def _setup_event_handlers(self):
        """Setup event handlers for broadcasting updates."""

        async def on_stats_update(event: Event):
            await self.manager.broadcast({
                "type": "stats",
                "data": event.data.get("stats", {})
            })

        async def on_project_change(event: Event):
            await self.manager.broadcast({
                "type": "project_change",
                "data": event.data
            })

        event_bus.subscribe_async(EventType.PROJECT_STATS_UPDATE, on_stats_update)
        event_bus.subscribe_async(EventType.PROJECT_START, on_project_change)
        event_bus.subscribe_async(EventType.PROJECT_STOP, on_project_change)

    async def _broadcast_stats_loop(self):
        """Periodically broadcast stats to all clients."""
        while True:
            await asyncio.sleep(0.5)  # 2 updates per second
            state = self.get_state()
            await self.manager.broadcast({
                "type": "state",
                "data": state
            })

    def run(self, host: str = None, port: int = None):
        """Run the server (blocking)."""
        uvicorn.run(
            self.app,
            host=host or config.web_host,
            port=port or config.web_port,
            log_level="warning"
        )

    async def run_async(self, host: str = None, port: int = None):
        """Run the server asynchronously."""
        config_obj = uvicorn.Config(
            self.app,
            host=host or config.web_host,
            port=port or config.web_port,
            log_level="warning"
        )
        server = uvicorn.Server(config_obj)

        # Start broadcast task
        self._broadcast_task = asyncio.create_task(self._broadcast_stats_loop())

        await server.serve()

        # Cancel broadcast task
        if self._broadcast_task:
            self._broadcast_task.cancel()
