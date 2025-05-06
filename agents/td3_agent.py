# agents/td3_agent.py

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# —————————————————————————————————————————————————————————————
#  Replay Buffer (same as your DQN one)
# —————————————————————————————————————————————————————————————
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
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
#  Networks
# —————————————————————————————————————————————————————————————
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        layers = []
        dims = [state_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [ nn.Linear(dims[i], dims[i+1]), nn.ReLU() ]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        x = self.net(x)
        return torch.tanh(self.out(x))  # assume actions in [-1,1]


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        # Q1
        layers1 = []
        dims1 = [state_dim + action_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers1 += [ nn.Linear(dims1[i], dims1[i+1]), nn.ReLU() ]
        self.q1 = nn.Sequential(*layers1, nn.Linear(hidden_dims[-1], 1))
        # Q2
        layers2 = []
        dims2 = [state_dim + action_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers2 += [ nn.Linear(dims2[i], dims2[i+1]), nn.ReLU() ]
        self.q2 = nn.Sequential(*layers2, nn.Linear(hidden_dims[-1], 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


# —————————————————————————————————————————————————————————————
#  TD3 Agent
# —————————————————————————————————————————————————————————————
class TD3Agent:
    def __init__(self,
                env,
                hidden_dims      = [64, 64],    # same as DQN
                actor_lr         = 1e-3,        # same as your DQN’s lr
                critic_lr        = 1e-3,
                gamma            = 0.99,        # same discount
                tau              = 0.005,       # soft-update like DQN’s target_update
                buffer_size      = 10_000,      # match your DQN’s replay size
                batch_size       = 64,          # same minibatch size
                policy_noise     = 0.02,        # small Gaussian noise
                noise_clip       = 0.05,        # clipped to your action range
                policy_freq      = 2,           # update actor every 2 critic updates
                device           = None):
        
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # dims
        obs0 = env.reset()
        state_dim  = obs0.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        # actor + target
        self.actor       = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt   = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # critic + target
        self.critic        = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt    = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # replay buffer
        self.buffer     = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # TD3 params
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip   = noise_clip * max_action
        self.policy_freq  = policy_freq
        self.max_action   = max_action

        self.total_it = 0

    def select_action(self, state):
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(st).cpu().data.numpy().flatten()

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.add((s, a, r, s_next, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        self.total_it += 1
        # sample
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        st   = torch.tensor(states,      device=self.device)
        ac   = torch.tensor(actions,     device=self.device)
        rw   = torch.tensor(rewards,     device=self.device)
        st2  = torch.tensor(next_states, device=self.device)
        done = torch.tensor(dones,       device=self.device)

        # ————————————————— Critic update —————————————————
        with torch.no_grad():
            # target action smoothing
            noise = (torch.randn_like(ac) * self.policy_noise)\
                    .clamp(-self.noise_clip, self.noise_clip)
            next_ac = (self.actor_target(st2) * self.max_action + noise)\
                      .clamp(-self.max_action, self.max_action)
            # target Q
            q1_t, q2_t = self.critic_target(st2, next_ac)
            q_target   = torch.min(q1_t, q2_t)
            target     = rw + (1 - done) * self.gamma * q_target

        q1, q2 = self.critic(st, ac)
        loss_critic = nn.MSELoss()(q1, target) + nn.MSELoss()(q2, target)

        self.critic_opt.zero_grad()
        loss_critic.backward()
        self.critic_opt.step()

        # ————————————————— Delayed actor + target updates —————————————————
        if self.total_it % self.policy_freq == 0:
            # actor loss (maximize Q)
            actor_loss = -self.critic.q1(torch.cat([st, self.actor(st)], 1)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # soft update targets
            for param, target in zip(self.actor.parameters(), self.actor_target.parameters()):
                target.data.mul_(1 - self.tau)
                target.data.add_(self.tau * param.data)
            for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
                target.data.mul_(1 - self.tau)
                target.data.add_(self.tau * param.data)

    def save(self, path):
        torch.save({
            'actor':       self.actor.state_dict(),
            'critic':      self.critic.state_dict(),
            'actor_target':self.actor_target.state_dict(),
            'critic_target':self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
