#!/usr/bin/env python3
"""
Display Computer - Main Entry Point

Runs the display launcher and web server together.
"""

import asyncio
import threading
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shared.config import config
from launcher.main import Launcher
from web.server import WebServer


def run_web_server(launcher: Launcher):
    """Run the web server in a separate thread."""
    server = WebServer(get_state=launcher.get_state)

    # Run in a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        server.run()
    except Exception as e:
        print(f"Web server error: {e}")
    finally:
        loop.close()


def main():
    parser = argparse.ArgumentParser(description="Display Computer")
    parser.add_argument(
        "--windowed", "-w",
        action="store_true",
        help="Run in windowed mode instead of fullscreen"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Screen width (default: 1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Screen height (default: 1080)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Web server port (default: 8000)"
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable web server"
    )

    args = parser.parse_args()

    # Apply config
    if args.windowed:
        config.fullscreen = False
    config.screen_width = args.width
    config.screen_height = args.height
    config.web_port = args.port

    print("=" * 50)
    print("  DISPLAY COMPUTER")
    print("=" * 50)
    print(f"  Resolution: {config.screen_width}x{config.screen_height}")
    print(f"  Fullscreen: {config.fullscreen}")
    print(f"  Web server: http://0.0.0.0:{config.web_port}")
    print("=" * 50)
    print()

    # Initialize launcher
    launcher = Launcher()
    launcher.init()

    # Start web server in background thread
    if not args.no_web:
        web_thread = threading.Thread(
            target=run_web_server,
            args=(launcher,),
            daemon=True
        )
        web_thread.start()
        print(f"Web server started on port {config.web_port}")

    # Run main display loop (blocking)
    try:
        launcher.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        print("Goodbye!")


if __name__ == "__main__":
    main()
