import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

# ─── Enhanced Networks ─────────────────────────────────────────────────────────

class DeepActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        # encoder
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # action head
        self.out = nn.Linear(hidden_dim, act_dim)

    def forward(self, x):
        h1 = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(self.dropout(h1))) + h1)
        h3 = F.relu(self.ln3(self.fc3(self.dropout(h2))) + h2)
        return torch.tanh(self.out(self.dropout(h3)))

class DeepCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        # encoder
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # value head
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = F.relu(self.ln1(self.fc1(x)))
        h2 = F.relu(self.ln2(self.fc2(self.dropout(h1))) + h1)
        h3 = F.relu(self.ln3(self.fc3(self.dropout(h2))) + h2)
        return self.value(self.dropout(h3))

# ─── PPO Agent ────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self, obs_dim, act_dim,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_ratio=0.2, epochs=10,
                 hidden_dim=256, dropout=0.1):
        self.gamma       = gamma
        self.lam         = lam
        self.clip_ratio  = clip_ratio
        self.epochs      = epochs

        # use the deep actor/critic
        self.actor  = DeepActor(obs_dim, act_dim, hidden_dim, dropout)
        self.critic = DeepCritic(obs_dim, hidden_dim, dropout)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = []

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)   # (1, obs_dim)
        mu    = self.actor(obs_t)                    # bounded in [-1,1]
        dist  = torch.distributions.Normal(mu, 1.0)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value    = self.critic(obs_t).squeeze(-1)

        # Clip action to safe limits in case your env needs it
        action = torch.clamp(
            action,
            min=torch.tensor([-0.1, -0.2*np.pi/4]),
            max=torch.tensor([ 0.1,  0.2*np.pi/4])
        )

        return action.squeeze(0).detach().numpy(), log_prob.item(), value.item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def clear_memory(self):
        self.memory = []

    def compute_returns_and_advantages(self):
        obs, actions, log_probs, rewards, dones, values = zip(*self.memory)

        advantages = []
        returns    = []
        gae = 0
        values = list(values) + [0.0]  # bootstrap

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1-dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return {
            "obs":        torch.FloatTensor(np.array(obs)),
            "actions":    torch.FloatTensor(np.array(actions)),
            "log_probs":  torch.FloatTensor(log_probs),
            "returns":    torch.FloatTensor(returns),
            "advantages": torch.FloatTensor(advantages),
        }

    def update(self):
        batch = self.compute_returns_and_advantages()
        adv   = batch['advantages']
        adv   = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.epochs):
            mu = self.actor(batch['obs'])
            dist = torch.distributions.Normal(mu, 1.0)
            new_log_probs = dist.log_prob(batch['actions']).sum(dim=-1)

            ratio        = torch.exp(new_log_probs - batch['log_probs'])
            clipped_ratio = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
            policy_loss  = -torch.min(ratio * adv, clipped_ratio * adv).mean()

            values       = self.critic(batch['obs']).squeeze(-1)
            value_loss   = (batch['returns'] - values).pow(2).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        print(f"Policy Loss: {policy_loss.item():.3f}, Value Loss: {value_loss.item():.3f}")
        self.clear_memory()

    def save(self, path):
        torch.save({
            'actor_state_dict':  self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, path)

    def select_action(self, obs, deterministic: bool = False):
        """
        Given an observation, returns:
          - action: np.array([v, ω])
          - log_prob: float (0 if deterministic=True)
          - value: float
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
        mu    = self.actor(obs_t)                   # network mean in [-1,1]

        if deterministic:
            action   = mu
            log_prob = None
        else:
            dist     = torch.distributions.Normal(mu, 1.0)
            action   = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        value = self.critic(obs_t).squeeze(-1)

        # Clip to your safe action limits
        action = torch.clamp(
            action,
            min=torch.tensor([-0.1, -0.2*np.pi/4]),
            max=torch.tensor([ 0.1,  0.2*np.pi/4])
        )

        lp = log_prob.item() if log_prob is not None else 0.0
        return action.squeeze(0).detach().numpy(), lp, value.item()

    def load(self, path):
        """
        Load actor & critic weights and set networks to eval mode.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor.eval()
        self.critic.eval()
