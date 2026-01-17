# Display Computer

A collection of visual projects for a living room display computer, featuring reinforcement learning agents, simulations, and interactive visualizations.

## Features

- **Launcher Menu** - Visual project selector with arcade-style interface
- **Web Control** - Control projects remotely from your phone
- **Extensible** - Easy to add new projects

## Projects

- **Flappy RL** - Watch a neural network learn to play Flappy Bird in real-time

## Tech Stack

- Python 3.11+
- Pygame (rendering)
- PyTorch (reinforcement learning)
- FastAPI (web control interface)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python run.py
```

## Headless / No-GUI Setup

For machines without a desktop environment (e.g., a dedicated display PC), the app automatically starts X11 when no display is detected.

**Install X11:**
```bash
# Arch-based (Manjaro, EndeavourOS, etc.)
sudo pacman -S xorg-server xorg-xinit

# Debian/Ubuntu
sudo apt install xorg xinit

# Fedora
sudo dnf install xorg-x11-server-Xorg xorg-x11-xinit
```

**Run:**
```bash
source venv/bin/activate
python run.py
```

The app detects there's no `DISPLAY` environment variable and launches X11 with itself as the only client. When the app exits, X11 exits automatically - perfect for kiosk-style setups.

**Auto-start on boot (optional):**

Create a systemd service or add to your shell profile:
```bash
# Example: ~/.bash_profile or ~/.zprofile
cd /path/to/display_computer && source venv/bin/activate && python run.py
```

## Web Interface

Once running, access the control panel at `http://<display-ip>:8000`

## Adding New Projects

1. Create a new directory in `projects/`
2. Implement the `BaseProject` interface
3. The launcher auto-discovers new projects on startup
