import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU()
            ]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.net(x)

class DQNAgent:
    def __init__(self,
                 env,
                 hidden_dims=[64,64],
                 lr=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 buffer_size=10_000,
                 target_update=10,
                 use_lidar: bool = False,
                 lidar_range: int = 10,
                 use_history: bool = False,
                 history_length: int = 3,
                 device=None):

        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- store LiDAR settings ----
        self.use_lidar   = use_lidar
        self.lidar_range = lidar_range

        # ---- store history settings ----
        self.use_history    = use_history
        self.history_length = history_length
        
        # ---- Dynamic input size ----
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            sample_obs, _ = reset_output
        else:
            sample_obs = reset_output
        self.input_dim = int(sample_obs.shape[0])
        self.output_dim = env.action_space.n

        self.output_dim = env.action_space.n

        # ---- build policy & target nets ----
        self.policy_net = DQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net = DQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # ---- optimizer, loss, discount ----
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma     = gamma

        # ---- ε-greedy ----
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # ---- replay buffer ----
        self.memory     = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        # ---- bookkeeping ----
        self.target_update = target_update
        self.step_counter  = 0

    def select_action(self, state) -> int:
        exploration_prob = random.random()
        if exploration_prob < self.epsilon:
            action = self.env.action_space.sample()
            return action
        else:
            state_tensor = torch.tensor(
                state, 
                dtype=torch.float32, 
                device=self.device
                ).unsqueeze(0) # unsqueeze to get shape (1, input_dim)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
            return action
            
    def store_transition(self, s, a, r, s_next, done):
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

    def optimize_model(self):
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

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def maybe_update_target(self):
        if self.step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str) -> None:
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str) -> None:
        state_dict = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
