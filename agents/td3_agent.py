import random
import numpy as np
from collections import deque
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# ————————————————
# Replay Buffer
# ————————————————
class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# ————————————————
# Flexible Networks
# ————————————————
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=32):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        layers = []
        input_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.net(x))

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=32):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        layers1 = []
        layers2 = []
        input_dim = state_dim + action_dim
        for h in hidden_dims:
            layers1 += [nn.Linear(input_dim, h), nn.ReLU()]
            layers2 += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers1.append(nn.Linear(input_dim, 1))
        layers2.append(nn.Linear(input_dim, 1))
        self.q1 = nn.Sequential(*layers1)
        self.q2 = nn.Sequential(*layers2)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

# ————————————————
# Juiced-down TD3 Agent with variable network size
# ————————————————
class TD3Agent:
    def __init__(
        self,
        env,
        hidden_dims=32,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=5000,
        batch_size=32,
        policy_noise=0.05,
        noise_clip=0.1,
        policy_freq=2,
        device=None
    ):
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        obs, *_ = env.reset()
        state_dim = obs.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        # actor
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # critic
        self.critic = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # replay
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.batch_size = batch_size

        # params
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq
        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state, noise: float = 0.0):
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        raw = self.actor(st).cpu().data.numpy().flatten()
        action = raw * self.max_action
        if noise:
            action += np.random.normal(0, noise * self.max_action, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.add((s, a, r, s_next, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        self.total_it += 1
        st, ac, rw, st2, dn = self.buffer.sample(self.batch_size)
        st, ac, rw, st2, dn = [t.to(self.device) for t in (st, ac, rw, st2, dn)]

        # target actions
        noise = (torch.randn_like(ac) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_ac = (self.actor_target(st2) * self.max_action + noise).clamp(-self.max_action, self.max_action)
        with torch.no_grad():
            q1_t, q2_t = self.critic_target(st2, next_ac)
            target = rw + (1 - dn) * self.gamma * torch.min(q1_t, q2_t)

        # critic update
        q1, q2 = self.critic(st, ac)
        loss_c = nn.MSELoss()(q1, target) + nn.MSELoss()(q2, target)
        self.critic_opt.zero_grad(); loss_c.backward(); self.critic_opt.step()

        # actor update
        if self.total_it % self.policy_freq == 0:
            act_pred = self.actor(st) * self.max_action
            q1_pred, _ = self.critic(st, act_pred)
            actor_loss = -q1_pred.mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()
            # soft update
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)

    def save(self, path: str):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])

def train_td3(env, agent, episodes: int = 500, max_steps: int = 1000, noise: float = 0.1):
    """
    Train a TD3 agent and show progress with reward per episode.
    Returns the list of episode rewards.
    """
    rewards = []
    pbar = tqdm(range(episodes), desc="TD3 Training")
    for ep in pbar:
        state, _ = env.reset()
        ep_r = 0
        for t in range(max_steps):
            a = agent.select_action(state, noise)
            nxt, r, done, _ = env.step(a)
            agent.store_transition(state, a, r, nxt, float(done))
            agent.train_step()
            state = nxt
            ep_r += r
            if done:
                break
        rewards.append(ep_r)
        pbar.set_postfix({"Reward": f"{ep_r:.1f}"})
    return rewards
