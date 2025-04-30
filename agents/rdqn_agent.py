# agents/rdqn_agent.py

import random
import numpy as np
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------------------------------
# 1) Recurrent Q-Network using LSTM
# --------------------------------------------------------------------------
class RDQNNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        input_dim: dimension of state vector (e.g. 2 or 6 with LiDAR)
        hidden_dim: size of LSTM hidden state
        output_dim: number of actions
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # one-layer LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # output head
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hx: tuple):
        """
        x: tensor of shape (batch, seq_len, input_dim)
        hx: tuple (h0, c0) each of shape (1, batch, hidden_dim)
        Returns:
          q: tensor of shape (batch, output_dim)
          hx_next: updated hidden state tuple
        """
        out_seq, hx_next = self.lstm(x, hx)
        # take last time-step's output for Q-values
        last = out_seq[:, -1, :]              # shape (batch, hidden_dim)
        q    = self.fc(last)                  # shape (batch, output_dim)
        return q, hx_next

# --------------------------------------------------------------------------
# 2) Recurrent DQN Agent
# --------------------------------------------------------------------------
class RDQNAgent:
    def __init__(self,
                 env,
                 hidden_dim: int = 64,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update: int = 10,
                 use_lidar: bool = False,
                 lidar_range: int = 10,
                 device: str = None):
        """
        A recurrent DQN agent using an LSTM to embed short-term history.
        """
        self.env = env
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma  = gamma

        # ε-greedy parameters
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # LiDAR flags (env controls obs shape)
        self.use_lidar   = use_lidar
        self.lidar_range = lidar_range

        # determine input and output dims
        obs0 = env.reset()
        self.input_dim  = int(np.array(obs0).shape[0])
        self.output_dim = env.action_space.n

        # networks
        self.policy_net = RDQNNetwork(self.input_dim, hidden_dim, self.output_dim).to(self.device)
        self.target_net = RDQNNetwork(self.input_dim, hidden_dim, self.output_dim).to(self.device)
        # sync parameters
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer & loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # for target updates
        self.target_update = target_update
        self.step_counter  = 0

    def select_action(self, state: np.ndarray, hx: tuple):
        """
        Choose an action given state and hidden state hx.
        Returns (action, hx_next).
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample(), hx

        # forward through LSTM
        st = torch.tensor(state, dtype=torch.float32, device=self.device)
        st = st.unsqueeze(0).unsqueeze(0)  # shape (1,1,input_dim)
        q_vals, hx_next = self.policy_net(st, hx)
        action = int(q_vals.argmax(dim=1).item())
        return action, hx_next

    def optimize_step(self, state, action, reward, next_state, done, hx, hx_next):
        """
        Perform one Q-learning update using the current and next hidden states.
        """
        # predict Q(s,a)
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        q_pred, _ = self.policy_net(st, hx)
        q_pred_val = q_pred[0, action]

        # compute target Q-value
        with torch.no_grad():
            nst = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            q_next, _ = self.target_net(nst, hx_next)
            max_next = q_next.max(dim=1)[0].item()
            q_target = reward + (0.0 if done else self.gamma * max_next)

        # loss and backward
        loss = self.criterion(q_pred_val, torch.tensor(q_target, device=self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network periodically
        self.step_counter += 1
        if self.step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """
        Save only the policy_net parameters; target_net can be rebuilt.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        sd = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(sd)
        self.target_net.load_state_dict(sd)

# --------------------------------------------------------------------------
# 3) Training loop for RDQN
# --------------------------------------------------------------------------
# utils/grid_utils.py (or your rdqn_agent module)

def train_rdqn(env,
               agent: RDQNAgent,
               episodes: int = 1000,
               max_steps: int = 100):
    """
    Train a recurrent DQN agent online (no replay), with a tqdm progress bar.
    Each episode resets the LSTM hidden state.
    """
    rewards_hist = []

    # wrap episodes in a tqdm progress bar
    pbar = tqdm(range(episodes), desc="R-DQN Training")
    for ep in pbar:
        state = env.reset()
        # initialize LSTM hidden/cell states
        h0 = torch.zeros(1, 1, agent.policy_net.hidden_dim, device=agent.device)
        c0 = torch.zeros(1, 1, agent.policy_net.hidden_dim, device=agent.device)
        hx = (h0, c0)

        total_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            action, hx_next = agent.select_action(state, hx)
            next_state, reward, done, _ = env.step(action)

            agent.optimize_step(state, action, reward, next_state, done, hx, hx_next)

            state = next_state
            # detach hidden state so backprop doesn’t span episodes
            hx = (hx_next[0].detach(), hx_next[1].detach())
            total_reward += reward
            step += 1

        agent.update_epsilon()
        rewards_hist.append(total_reward)

        # update the progress bar postfix to show current reward & epsilon
        pbar.set_postfix({
            "Reward": f"{total_reward:.1f}",
            "ε":      f"{agent.epsilon:.3f}"
        })

    return rewards_hist
