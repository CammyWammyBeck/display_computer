"""DQN Agent for Flappy Bird."""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import flappy_config as cfg


class DQN(nn.Module):
    """Deep Q-Network."""

    def __init__(self, input_size: int = 4, output_size: int = 2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.
    Each leaf holds a priority, and we can sample proportionally in O(log n).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find the leaf index for a given cumulative value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data):
        """Add data with given priority."""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update priority at tree index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, value: float) -> tuple[int, float, object]:
        """Get leaf index, priority, and data for cumulative value."""
        idx = self._retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Samples transitions with probability proportional to their TD error.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent (0 = uniform, 1 = full prioritization)
        self.beta_start = beta_start  # Importance sampling start
        self.beta_frames = beta_frames
        self.frame = 0
        self.max_priority = 1.0
        self.min_priority = 0.01

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience with max priority (will be updated after training)."""
        data = (state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, data)

    def sample(self, batch_size: int) -> tuple:
        """Sample a prioritized batch."""
        self.frame += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

        batch = []
        indices = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = random.uniform(low, high)
            idx, priority, data = self.tree.get(value)

            if data is None or (isinstance(data, (int, float)) and data == 0):
                # Handle edge case of empty slot
                value = random.uniform(0, self.tree.total())
                idx, priority, data = self.tree.get(value)

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities) + 1e-6
        probs = priorities / self.tree.total()
        weights = (self.tree.size * probs) ** (-beta)
        weights = weights / weights.max()

        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: list, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.size


class DQNAgent:
    """
    DQN Agent for playing Flappy Bird.

    Uses experience replay and a target network for stable learning.
    """

    def __init__(self, save_dir: Path = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)

        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(cfg.memory_size)

        # Exploration
        self.epsilon = cfg.epsilon_start

        # Training stats
        self.episode = 0
        self.total_steps = 0
        self.best_score = 0
        self.recent_scores: deque = deque(maxlen=100)
        self.losses: deque = deque(maxlen=100)

        # Full history for graphs (limited to last N episodes)
        self.max_history = cfg.graph_history_length
        self.score_history: list[int] = []
        self.avg_history: list[float] = []
        self.epsilon_history: list[float] = []
        self.loss_history: list[float] = []

        # Save directory
        self.save_dir = save_dir or Path("models/flappy_rl")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current observation
            training: If True, use exploration. If False, be greedy.

        Returns:
            Action (0 = nothing, 1 = flap)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store a transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self) -> float | None:
        """
        Perform one training step with Double DQN and Prioritized Experience Replay.

        Returns:
            Loss value or None if not enough samples
        """
        if len(self.memory) < cfg.batch_size:
            return None

        # Sample prioritized batch
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(cfg.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q values using Double DQN
        # Use policy net to select actions, target net to evaluate them
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * cfg.gamma * next_q

        # Compute TD errors for priority updates
        td_errors = (current_q - target_q).detach().cpu().numpy()

        # Compute weighted loss (importance sampling correction)
        element_wise_loss = (current_q - target_q) ** 2
        loss = (weights * element_wise_loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def end_episode(self, score: int):
        """Called at the end of each episode."""
        self.episode += 1
        self.recent_scores.append(score)

        # Record history for graphs
        self.score_history.append(score)
        avg_score = np.mean(self.recent_scores)
        self.avg_history.append(avg_score)
        self.epsilon_history.append(self.epsilon)
        avg_loss = np.mean(self.losses) if self.losses else 0
        self.loss_history.append(avg_loss)

        # Trim history to max length
        if len(self.score_history) > self.max_history:
            self.score_history = self.score_history[-self.max_history:]
            self.avg_history = self.avg_history[-self.max_history:]
            self.epsilon_history = self.epsilon_history[-self.max_history:]
            self.loss_history = self.loss_history[-self.max_history:]

        if score > self.best_score:
            self.best_score = score
            self.save("best")

        # Decay epsilon
        self.epsilon = max(cfg.epsilon_end, self.epsilon * cfg.epsilon_decay)

        # Update target network
        if self.episode % cfg.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_stats(self) -> dict:
        """Get current training statistics."""
        avg_score = np.mean(self.recent_scores) if self.recent_scores else 0
        avg_loss = np.mean(self.losses) if self.losses else 0

        return {
            "episode": self.episode,
            "epsilon": round(self.epsilon, 4),
            "best_score": self.best_score,
            "avg_score_100": round(avg_score, 2),
            "avg_loss": round(avg_loss, 6),
            "memory_size": len(self.memory),
            "total_steps": self.total_steps,
        }

    def save(self, name: str = "checkpoint"):
        """Save model and training state."""
        path = self.save_dir / f"{name}.pt"
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode": self.episode,
            "best_score": self.best_score,
            "total_steps": self.total_steps,
            # History for graphs
            "score_history": self.score_history,
            "avg_history": self.avg_history,
            "epsilon_history": self.epsilon_history,
            "loss_history": self.loss_history,
            "recent_scores": list(self.recent_scores),
        }, path)
        print(f"Saved model to {path}")

    def load(self, name: str = "checkpoint") -> bool:
        """Load model and training state."""
        path = self.save_dir / f"{name}.pt"
        if not path.exists():
            print(f"No checkpoint found at {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = checkpoint["epsilon"]
            self.episode = checkpoint["episode"]
            self.best_score = checkpoint["best_score"]
            self.total_steps = checkpoint["total_steps"]

            # Load history if available (backwards compatibility)
            self.score_history = checkpoint.get("score_history", [])
            self.avg_history = checkpoint.get("avg_history", [])
            self.epsilon_history = checkpoint.get("epsilon_history", [])
            self.loss_history = checkpoint.get("loss_history", [])

            recent = checkpoint.get("recent_scores", [])
            self.recent_scores = deque(recent, maxlen=100)

            print(f"Loaded model from {path} (episode {self.episode}, best: {self.best_score})")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def get_history(self) -> dict:
        """Get training history for graph restoration."""
        return {
            "scores": self.score_history.copy(),
            "averages": self.avg_history.copy(),
            "epsilons": self.epsilon_history.copy(),
            "losses": self.loss_history.copy(),
        }

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for visualization."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]

    def get_network_activations(self, state: np.ndarray) -> list[np.ndarray]:
        """Get intermediate activations for visualization."""
        activations = []

        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            for layer in self.policy_net.network:
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    activations.append(x.cpu().numpy()[0])

        return activations
