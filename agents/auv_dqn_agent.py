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
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SimpleAuvDQNAgent:
    def __init__(self,
                 env,
                 hidden_dims=[128,128],
                 lr=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 batch_size=64,
                 buffer_size=10_000,
                 target_update=10):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # sample one obs to get dimension
        obs0 = env.reset()
        self.input_dim  = obs0.shape[0]
        self.output_dim = len(env.actions) if env.discrete_actions else env.action_space.n

        # nets
        self.policy_net = DQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net = DQNNetwork(self.input_dim, hidden_dims, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma     = gamma

        # ε-greedy
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # replay buffer
        self.memory     = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        # sync freq
        self.target_update = target_update
        self.step_counter  = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.output_dim)
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qv = self.policy_net(st)
        return int(qv.argmax(dim=1).item())

    def store_transition(self, s, a, r, s_next, done):
        self.memory.append((s,a,r,s_next,done))

    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor      = torch.tensor(np.array(states),      dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_tensor     = torch.tensor(actions,                dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor     = torch.tensor(rewards,                dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_tensor       = torch.tensor(dones,                  dtype=torch.float32, device=self.device).unsqueeze(1)
        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.sample_batch()
        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)
        # target: r + γ max_a' Q_target(s',a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q   = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # sync target
        self.step_counter += 1
        if self.step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            "state_dict": self.policy_net.state_dict(),
            "epsilon":    self.epsilon,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["state_dict"])
        self.target_net.load_state_dict(ckpt["state_dict"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
