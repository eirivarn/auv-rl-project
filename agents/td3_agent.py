import random
import pickle
from collections import deque
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange


# ─── Default TD3 Hyperparameters ───────────────────────────────────────────────
DEFAULT_HIDDEN_DIMS    = [64, 64]
DEFAULT_ACTOR_LR       = 1e-3
DEFAULT_CRITIC_LR      = 1e-3
DEFAULT_GAMMA          = 0.99
DEFAULT_TAU            = 0.005
DEFAULT_BUFFER_SIZE    = 20_000
DEFAULT_REPLAUY_BUFFER_SIZE = 5000
DEFAULT_BATCH_SIZE     = 32
DEFAULT_POLICY_NOISE   = 0.05
DEFAULT_NOISE_CLIP     = 0.1
DEFAULT_POLICY_FREQ    = 2
DEFAULT_TRAIN_EPISODES = 500
DEFAULT_TRAIN_MAX_STEPS = 1000
# ────────────────────────────────────────────────────────────────────────────────


class ReplayBuffer:
    def __init__(self, capacity: int = DEFAULT_REPLAUY_BUFFER_SIZE):
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor      = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor     = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards_tensor     = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_tensor       = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = DEFAULT_HIDDEN_DIMS
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        layers: List[nn.Module] = []
        input_dim = state_dim

        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h

        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = DEFAULT_HIDDEN_DIMS
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Q1 architecture
        layers_q1: List[nn.Module] = []
        # Q2 architecture
        layers_q2: List[nn.Module] = []

        input_dim = state_dim + action_dim
        for h in hidden_dims:
            layers_q1.append(nn.Linear(input_dim, h))
            layers_q1.append(nn.ReLU())
            layers_q2.append(nn.Linear(input_dim, h))
            layers_q2.append(nn.ReLU())
            input_dim = h

        layers_q1.append(nn.Linear(input_dim, 1))
        layers_q2.append(nn.Linear(input_dim, 1))

        self.q1 = nn.Sequential(*layers_q1)
        self.q2 = nn.Sequential(*layers_q2)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


class TD3Agent:
    def __init__(
        self,
        env,
        hidden_dims: List[int]       = DEFAULT_HIDDEN_DIMS,
        actor_lr: float              = DEFAULT_ACTOR_LR,
        critic_lr: float             = DEFAULT_CRITIC_LR,
        gamma: float                 = DEFAULT_GAMMA,
        tau: float                   = DEFAULT_TAU,
        buffer_size: int             = DEFAULT_BUFFER_SIZE,
        batch_size: int              = DEFAULT_BATCH_SIZE,
        policy_noise: float          = DEFAULT_POLICY_NOISE,
        noise_clip: float            = DEFAULT_NOISE_CLIP,
        policy_freq: int             = DEFAULT_POLICY_FREQ,
        device: Optional[str]        = None
    ):
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        obs, *_ = env.reset()
        state_dim  = obs.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size

        self.gamma        = gamma
        self.tau          = tau
        self.max_action   = max_action
        self.policy_noise = policy_noise * max_action
        self.noise_clip   = noise_clip * max_action
        self.policy_freq  = policy_freq

        self.total_steps = 0

    def select_action(
        self,
        state: np.ndarray,
        noise: float = 0.0
    ) -> np.ndarray:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_raw = self.actor(state_tensor).cpu().numpy().flatten()
        action = action_raw * self.max_action

        if noise > 0:
            noise_sample = np.random.normal(0, noise * self.max_action, size=action.shape)
            action = action + noise_sample

        return action.clip(-self.max_action, self.max_action)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> None:
        if len(self.replay_buffer) < self.batch_size:
            return

        self.total_steps += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        noise = (
            torch.randn_like(actions) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (
            self.actor_target(next_states) * self.max_action + noise
        ).clamp(-self.max_action, self.max_action)

        with torch.no_grad():
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target_min = torch.min(q1_target, q2_target)
            target_q = rewards + (1 - dones) * self.gamma * q_target_min

        q1_current, q2_current = self.critic(states, actions)
        loss_critic = nn.MSELoss()(q1_current, target_q) + nn.MSELoss()(q2_current, target_q)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        if self.total_steps % self.policy_freq == 0:

            actions_pred = self.actor(states) * self.max_action
            q1_pred, _   = self.critic(states, actions_pred)
            loss_actor   = -q1_pred.mean()

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

    def save(self, filepath: str) -> None:
        torch.save({
            "actor_state_dict":  self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict()
        }, filepath)

    def load(self, filepath: str) -> None:

        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])

    def train(
        self,
        episodes: int = DEFAULT_TRAIN_EPISODES,
        max_steps: int = DEFAULT_TRAIN_MAX_STEPS,
        noise: float = DEFAULT_POLICY_NOISE
    ) -> List[float]:

        rewards_history: List[float] = []
        pbar = trange(episodes, desc="TD3 Training")

        for ep in pbar:
            state, _ = self.env.reset()
            episode_reward = 0.0

            for t in range(max_steps):
                action = self.select_action(state, noise)
                next_state, reward, done, _ = self.env.step(action)

                self.store_transition(state, action, reward, next_state, float(done))
                self.train_step()

                state = next_state
                episode_reward += reward

                if done:
                    break

            rewards_history.append(episode_reward)
            pbar.set_postfix({"Reward": f"{episode_reward:.1f}"})

        return rewards_history
