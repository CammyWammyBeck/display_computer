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

## Web Interface

Once running, access the control panel at `http://<display-ip>:8000`

## Adding New Projects

1. Create a new directory in `projects/`
2. Implement the `BaseProject` interface
3. The launcher auto-discovers new projects on startup
