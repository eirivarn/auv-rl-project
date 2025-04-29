import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# 1) A small MLP to approximate Q(s,a)
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

    def forward(self, x):
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
                 device=None):
        """
        env: your GridDockEnv
        hidden_dims: list of hidden-layer sizes
        lr, gamma: learning rate & discount
        epsilon_*: for ε-greedy
        batch_size, buffer_size: replay‐buffer params
        target_update: sync freq for target network
        """
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # state is a 2-vector (dx,dy)
        self.input_dim  = len(env.observation_space.low)
        self.output_dim = env.action_space.n

        # policy & target nets
        self.policy_net = DQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net = DQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma     = gamma

        # ε-greedy params
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # replay buffer
        self.memory     = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        # bookkeeping
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
        self.memory.append((s, a, r, s_next, done))

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack into single numpy arrays
        states_np      = np.array(states,      dtype=np.float32)  # shape: [batch, state_dim]
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np     = np.array(actions,     dtype=np.int64)    # shape: [batch]
        rewards_np     = np.array(rewards,     dtype=np.float32)  # shape: [batch]
        dones_np       = np.array(dones,       dtype=np.float32)  # shape: [batch]

        # Convert once to torch tensors
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

        # current Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # target: r + γ * max_a' Q_target(s',a')
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

    def save(self, path: str, extra: dict = None):
        """
        Save a full checkpoint containing:
        - policy network state_dict
        - agent hyperparameters (e.g. hidden_dims, batch_size, etc.)
        - optionally any extra info you pass in
        """
        ckpt = {
            "state_dict": self.policy_net.state_dict(),
            "hidden_dims": self.policy_net.net[0].out_features,  # or store your config explicitly
            "batch_size": self.batch_size,
            # … add anything else you need …
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_from_checkpoint(cls, env, path: str, device=None):
        """
        Construct a new agent from a saved checkpoint.
        """
        ckpt = torch.load(path, map_location=device)
        # read architecture
        hidden_dims = ckpt["hidden_dims"]
        # instantiate agent with the same settings
        agent = cls(env,
                    hidden_dims=[hidden_dims, hidden_dims],  # or reconstruct full list
                    lr=ckpt.get("lr", 1e-3),
                    gamma=ckpt.get("gamma", 0.99),
                    epsilon_start=ckpt.get("epsilon_start", 1.0),
                    epsilon_min=ckpt.get("epsilon_min", 0.01),
                    epsilon_decay=ckpt.get("epsilon_decay", 0.995),
                    batch_size=ckpt.get("batch_size", 64),
                    buffer_size=ckpt.get("buffer_size", 10_000),
                    target_update=ckpt.get("target_update", 10),
                    device=device)
        # load weights
        agent.policy_net.load_state_dict(ckpt["state_dict"])
        agent.target_net.load_state_dict(ckpt["state_dict"])
        return agent
