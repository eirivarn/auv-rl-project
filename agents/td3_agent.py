import random
import numpy as np
import time
from collections import deque, namedtuple
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

# —————————————————————————————————————————————————————————————
#  Replay Buffer (unchanged)
# —————————————————————————————————————————————————————————————
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)
        self.transition = namedtuple('Transition', ['state','action','reward','next_state','done'])

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.float32),
            np.array(rewards,     dtype=np.float32).reshape(-1,1),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32).reshape(-1,1)
        )

    def __len__(self):
        return len(self.buffer)

# —————————————————————————————————————————————————————————————
#  Enhanced Networks: 3-layer MLP + LayerNorm + Dropout + Orthogonal Init
# —————————————————————————————————————————————————————————————
class EnhancedActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        assert len(hidden_dims) == 3, "hidden_dims must be length 3"
        self.net = nn.Sequential(
            nn.Linear(state_dim,     hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]), nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dims[2], action_dim)
        # orthogonal init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.01); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.tanh(self.out(x))

class EnhancedCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]), nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]), nn.ReLU(), nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]), nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        # orthogonal init
        for net in (self.q1, self.q2):
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


class TD3Agent:
    def __init__(
        self,
        env,
        hidden_dims    = [128,128,64],
        actor_lr       = 3e-4,
        critic_lr      = 3e-4,
        gamma          = 0.99,
        tau            = 0.005,
        buffer_size    = 100_000,
        batch_size     = 128,
        policy_noise   = 0.2,
        noise_clip     = 0.5,
        policy_freq    = 2,
        device         = None
    ):
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # dims and action bounds
        obs0 = env.reset()
        state_dim = obs0[0].shape[0] if isinstance(obs0, tuple) else obs0.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        # actor + target
        self.actor        = EnhancedActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = EnhancedActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt    = optim.Adam(self.actor.parameters(), lr=actor_lr)
        # critic + target
        self.critic        = EnhancedCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = EnhancedCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt    = optim.Adam(self.critic.parameters(), lr=critic_lr)
        # buffer
        self.buffer       = ReplayBuffer(capacity=buffer_size)
        self.batch_size   = batch_size
        # td3 params
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip   = noise_clip * max_action
        self.policy_freq  = policy_freq
        self.max_action   = max_action
        self.total_it     = 0

    def select_action(self, state, noise: float = 0.0):
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        raw = self.actor(st).cpu().data.numpy().flatten()
        action = raw * self.max_action
        if noise:
            action = action + np.random.normal(0, noise * self.max_action, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.add((s, a, r, s_next, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        self.total_it += 1
        # sample
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        st   = torch.tensor(states,       dtype=torch.float32, device=self.device)
        ac   = torch.tensor(actions,      dtype=torch.float32, device=self.device)
        rw   = torch.tensor(rewards,      dtype=torch.float32, device=self.device)
        st2  = torch.tensor(next_states,  dtype=torch.float32, device=self.device)
        dn   = torch.tensor(dones,        dtype=torch.float32, device=self.device)
        # critic update with clipped noise target
        with torch.no_grad():
            noise = (torch.randn_like(ac) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_ac = (self.actor_target(st2) * self.max_action + noise).clamp(-self.max_action, self.max_action)
            q1_t, q2_t = self.critic_target(st2, next_ac)
            q_min     = torch.min(q1_t, q2_t)
            target    = rw + (1 - dn) * self.gamma * q_min
        q1, q2 = self.critic(st, ac)
        critic_loss = nn.MSELoss()(q1, target) + nn.MSELoss()(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        # delayed actor + target update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1(torch.cat([st, self.actor(st)*self.max_action], dim=1)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            # soft updates
            for p, tp in zip(self.actor.parameters(),       self.actor_target.parameters()):
                tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)
            for p, tp in zip(self.critic.parameters(),      self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau); tp.data.add_(self.tau * p.data)

    def save(self, path: str):
        torch.save({
            'actor'        : self.actor.state_dict(),
            'critic'       : self.critic.state_dict(),
            'actor_target' : self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic_target.load_state_dict(ckpt['critic_target'])

# —————————————————————————————————————————————————————————————
#  Training Loop with tqdm
# —————————————————————————————————————————————————————————————
def train_td3(env, agent, episodes: int = 1000, max_steps: int = 1000, noise: float = 0.1):
    rewards = []
    tbar = trange(episodes, desc="Enhanced TD3")
    for ep in tbar:
        start = time.time()
        state, _ = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state, noise)
            nxt, r, done, _ = env.step(action)
            agent.store_transition(state, action, r, nxt, float(done))
            agent.train_step()
            state = nxt
            ep_reward += r
            if done:
                break
        rewards.append(ep_reward)
        it_per_s = (t+1)/(time.time()-start) if ep_reward else 0
        tbar.set_postfix({"Reward":f"{ep_reward:.2f}"})
    return rewards