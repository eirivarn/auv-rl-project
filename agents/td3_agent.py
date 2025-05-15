import random
import numpy as np
from collections import deque, namedtuple
import time
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, max_action):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * torch.tanh(self.net(x))


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super().__init__()
        # Q1
        layers1 = []
        dims1 = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers1 += [nn.Linear(dims1[i], dims1[i+1]), nn.ReLU()]
        layers1.append(nn.Linear(dims1[-1], output_dim))
        self.q1 = nn.Sequential(*layers1)
        # Q2
        layers2 = []
        dims2 = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers2 += [nn.Linear(dims2[i], dims2[i+1]), nn.ReLU()]
        layers2.append(nn.Linear(dims2[-1], output_dim))
        self.q2 = nn.Sequential(*layers2)

    def forward(self, x, a):
        xu = torch.cat([x, a], dim=1)
        return self.q1(xu), self.q2(xu)


class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)
        self.transition = namedtuple('Transition', ['state','action','reward','next_state','done'])

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(self.transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)


class TD3Agent:
    def __init__(
        self,
        env,
        hidden_dims=[256,256],
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        max_action=1.0,
        batch_size=100,
        buffer_size=1_000_000,
        device=None
    ):
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obs_dim = env.reset()[0].shape[0]
        act_dim = env.action_space.shape[0]
        self.max_action = max_action
        # Networks
        self.actor = Actor(obs_dim, hidden_dims, act_dim, max_action).to(self.device)
        self.actor_target = Actor(obs_dim, hidden_dims, act_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(obs_dim+act_dim, hidden_dims).to(self.device)
        self.critic_target = Critic(obs_dim+act_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0.0:
            action = (action + np.random.normal(0, noise, size=action.shape))
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, iterations=1):
        for _ in range(iterations):
            if len(self.replay_buffer) < self.batch_size:
                return
            self.total_it += 1
            # Sample batch
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

            # Add noise to action for target policy
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute target Q-value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

            # Critic update
            current_q1, current_q2 = self.critic(state, action)
            critic_loss = nn.MSELoss()(current_q1, target_q.detach()) + nn.MSELoss()(current_q2, target_q.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed actor update
            if self.total_it % self.policy_freq == 0:
                # actor loss: maximize Q by minimizing -Q
                q1_new, _ = self.critic(state, self.actor(state))
                actor_loss = -q1_new.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update targets for critic and actor
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save(self, prefix):
        torch.save(self.actor.state_dict(), f"{prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{prefix}_critic.pth")

    def load(self, prefix):
        self.actor.load_state_dict(torch.load(f"{prefix}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{prefix}_critic.pth"))


def train_td3(env, agent, episodes=1000, max_steps=1000, noise=0.1):
    reward_hist = []
    tbar = trange(episodes, desc="TD3 Training")
    for ep in tbar:
        start = time.time()
        state, _ = env.reset()
        ep_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state, noise)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, float(done))
            agent.train(iterations=1)
            state = next_state
            ep_reward += reward
            if done:
                break
        reward_hist.append(ep_reward)
        elapsed = time.time() - start
        it_per_s = (t+1) / elapsed if elapsed > 0 else 0
        tbar.set_postfix({
            "Ep Reward": f"{ep_reward:.2f}"
        })
    return reward_hist
