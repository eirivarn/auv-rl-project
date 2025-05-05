import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DuelingDQNNetwork, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU()
            ]
        self.feature = nn.Sequential(*layers)
        self.value_layer = nn.Linear(hidden_dims[-1], 1)
        self.advantage_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.feature(x)
        value = self.value_layer(x)                       # (batch, 1)
        advantage = self.advantage_layer(x)               # (batch, output_dim)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

# Standard replay buffer using deque
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
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

# Dueling DQN agent
class DuelingDQNAgent:
    def __init__(self,
                 env,
                 hidden_dims=[64,64],
                 lr=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 buffer_size=10000,
                 target_update=10,
                 device=None):
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # dimensions
        obs0 = env.reset()
        self.input_dim  = int(obs0.shape[0])
        self.output_dim = env.action_space.n

        # networks
        self.policy_net = DuelingDQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net = DuelingDQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma     = gamma

        # epsilon-greedy parameters
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # replay buffer
        self.buffer     = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # target network update frequency
        self.target_update = target_update
        self.step_counter  = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.policy_net(st)
            return int(qv.argmax(dim=1).item())

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.add((s, a, r, s_next, done))

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        # to tensors
        states_tensor      = torch.from_numpy(states).to(self.device)
        actions_tensor     = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_tensor     = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states_tensor = torch.from_numpy(next_states).to(self.device)
        dones_tensor       = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        # current Q values
        q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        # double DQN target computation
        with torch.no_grad():
            next_actions = self.policy_net(next_states_tensor).argmax(1, keepdim=True)
            next_q       = self.target_net(next_states_tensor).gather(1, next_actions)
            target_q     = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

        # compute loss and backprop
        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def maybe_update_target(self):
        self.step_counter += 1
        if self.step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            "state_dict": self.policy_net.state_dict(),
            "epsilon": self.epsilon
        }, path)

    @classmethod
    def load_from_checkpoint(cls, env, path, hidden_dims=[64,64], device=None):
        ckpt = torch.load(path, map_location=device)
        agent = cls(env, hidden_dims=hidden_dims, device=device)
        agent.policy_net.load_state_dict(ckpt["state_dict"])
        agent.target_net.load_state_dict(ckpt["state_dict"])
        agent.epsilon = ckpt.get("epsilon", agent.epsilon)
        return agent



