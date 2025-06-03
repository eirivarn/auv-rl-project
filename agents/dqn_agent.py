import random
import pickle
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

# ─── Default DQN Hyperparameters ────────────────────────────────────────────────
DEFAULT_HIDDEN_DIMS     = [64, 64]
DEFAULT_LR              = 1e-3
DEFAULT_GAMMA           = 0.99
DEFAULT_EPSILON_START   = 1.0
DEFAULT_EPSILON_END     = 0.01
DEFAULT_EPSILON_DECAY   = 0.995
DEFAULT_BATCH_SIZE      = 64
DEFAULT_BUFFER_SIZE     = 10_000
DEFAULT_TARGET_UPDATE   = 10
DEFAULT_TRAIN_EPISODES  = 1000
DEFAULT_TRAIN_MAX_STEPS = 100
# ────────────────────────────────────────────────────────────────────────────────


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        env,
        hidden_dims: list[int]      = DEFAULT_HIDDEN_DIMS,
        lr: float                   = DEFAULT_LR,
        gamma: float                = DEFAULT_GAMMA,
        epsilon_start: float        = DEFAULT_EPSILON_START,
        epsilon_min: float          = DEFAULT_EPSILON_END,
        epsilon_decay: float        = DEFAULT_EPSILON_DECAY,
        batch_size: int             = DEFAULT_BATCH_SIZE,
        buffer_size: int            = DEFAULT_BUFFER_SIZE,
        target_update: int          = DEFAULT_TARGET_UPDATE,
        device: Optional[str]       = None,
    ):
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            sample_obs, _ = reset_out
        else:
            sample_obs = reset_out
        self.input_dim  = int(sample_obs.shape[0])
        self.output_dim = env.action_space.n

        self.policy_net = DQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net = DQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory     = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.target_update = target_update
        self.step_counter  = 0

        self.rewards_history: list[float] = []

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item()
        return action

    def store_transition(self, s, a, r, s_next, done) -> None:
        self.memory.append((s, a, r, s_next, done))

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_np      = np.array(states,      dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np     = np.array(actions,     dtype=np.int64)
        rewards_np     = np.array(rewards,     dtype=np.float32)
        dones_np       = np.array(dones,       dtype=np.float32)

        states_tensor      = torch.from_numpy(states_np).to(self.device)
        next_states_tensor = torch.from_numpy(next_states_np).to(self.device)
        actions_tensor     = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)
        rewards_tensor     = torch.from_numpy(rewards_np).unsqueeze(1).to(self.device)
        dones_tensor       = torch.from_numpy(dones_np).unsqueeze(1).to(self.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def optimize_model(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q   = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def maybe_update_target(self) -> None:
        if self.step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump({
                "policy_state_dict": self.policy_net.state_dict(),
                "epsilon": self.epsilon,
            }, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            ckpt = pickle.load(f)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["policy_state_dict"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)

    def train(self,
              episodes: int = DEFAULT_TRAIN_EPISODES,
              max_steps: int = DEFAULT_TRAIN_MAX_STEPS) -> list[float]:
        
        episode_returns: list[float] = []
        for ep in trange(episodes, desc="DQN Training"):
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple):
                state, _ = reset_out
            else:
                state = reset_out

            total_reward = 0.0
            for t in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.store_transition(state, action, reward, next_state, done)
                self.optimize_model()

                state = next_state
                total_reward += reward

                self.step_counter += 1
                self.maybe_update_target()

                if done:
                    break

            self.update_epsilon()
            episode_returns.append(total_reward)

        self.rewards_history = episode_returns
        return episode_returns

    def evaluate(self,
                 episodes: int = 100,
                 max_steps: int = 100,
                 render: bool = False) -> float:
        successes = 0
        for ep in range(episodes):
            obs_out = self.env.reset()
            if isinstance(obs_out, tuple):
                obs, _ = obs_out
            else:
                obs = obs_out

            for t in range(max_steps):
                state_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.policy_net(state_tensor).argmax(dim=1).item()

                next_obs, reward, done, _ = self.env.step(action)
                obs = next_obs

                if render:
                    self.env.render()

                if done:
                    successes += 1
                    break

        return successes / episodes
