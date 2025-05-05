import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# 1) Dueling DQN Network
# ---------------------------
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        # shared feature layers
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        self.feature = nn.Sequential(*layers)
        # value stream
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1)
        )
        # advantage stream
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        # combine streams: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

# ---------------------------
# 2) Prioritized Replay Buffer
# ---------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.sum_tree = np.zeros(2*capacity - 1)
        self.data = [None] * capacity
        self.max_priority = 1.0
        self.n_entries = 0

    def add(self, transition):
        idx = self.pos + self.capacity - 1
        self.data[self.pos] = transition
        self.update(idx, self.max_priority)
        self.pos = (self.pos + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority**self.alpha - self.sum_tree[idx]
        self.sum_tree[idx] += change
        # propagate change up
        while idx != 0:
            idx = (idx - 1) // 2
            self.sum_tree[idx] += change

    def _retrieve(self, idx, s):
        left = 2*idx + 1
        right = left + 1
        if left >= len(self.sum_tree):
            return idx
        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])

    def sample(self, batch_size, beta=0.4):
        indices = []
        transitions = []
        priorities = []
        segment = self.sum_tree[0] / batch_size
        for i in range(batch_size):
            s = random.uniform(segment*i, segment*(i+1))
            idx = self._retrieve(0, s)
            data_idx = idx - (self.capacity - 1)
            transitions.append(self.data[data_idx])
            priorities.append(self.sum_tree[idx])
            indices.append(idx)
        sampling_probabilities = np.array(priorities) / self.sum_tree[0]
        is_weights = np.power(self.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        return transitions, indices, is_weights

    def __len__(self):
        return self.n_entries

# ---------------------------
# 3) Enhanced DQN Agent
# ---------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class EnhancedDQNAgent:
    def __init__(
        self,
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
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
        device=None
    ):
        self.env = env
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        obs0 = env.reset()
        self.input_dim  = int(obs0.shape[0])
        self.n_actions  = env.action_space.n

        # networks
        self.policy_net = DuelingDQNNetwork(self.input_dim, hidden_dims, self.n_actions).to(self.device)
        self.target_net = DuelingDQNNetwork(self.input_dim, hidden_dims, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer & loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma     = gamma

        # epsilon
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_frames= beta_frames
        self.frame_idx  = 0

        # target update freq
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
        self.buffer.add(Transition(s, a, r, s_next, done))

    def update_beta(self):
        self.frame_idx += 1
        return min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def optimize_model(self):
        if len(self.buffer) < self.batch_size:
            return
        beta = self.update_beta()
        transitions, indices, is_weights = self.buffer.sample(self.batch_size, beta)
        batch = Transition(*zip(*transitions))

        states      = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards     = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        is_weights  = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # current Q
        q_values = self.policy_net(states).gather(1, actions)
        # double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # compute loss with importance-sampling weights
        loss = (q_values - target_q).pow(2) * is_weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities
        for idx, prio in zip(indices, prios.detach().cpu().numpy()):
            self.buffer.update(idx, prio)

        # epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # maybe update target
        self.step_counter += 1
        if self.step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            'state_dict': self.policy_net.state_dict(),
            'epsilon': self.epsilon
        }, path)

    @classmethod
    def load_from_checkpoint(cls, env, path, hidden_dims, device=None, **kwargs):
        ckpt = torch.load(path, map_location=device)
        agent = cls(env, hidden_dims=hidden_dims, device=device, **kwargs)
        agent.policy_net.load_state_dict(ckpt['state_dict'])
        agent.target_net.load_state_dict(ckpt['state_dict'])
        agent.epsilon = ckpt.get('epsilon', agent.epsilon)
        return agent
