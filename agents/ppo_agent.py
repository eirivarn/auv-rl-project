import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, epochs=10):
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )


        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = []

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)  # (1, obs_dim)
        logits = self.actor(obs)
        dist = torch.distributions.Normal(logits, 1.0)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action.squeeze(0).detach().numpy(), log_prob.item(), value.item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def clear_memory(self):
        self.memory = []

    def compute_returns_and_advantages(self):
        obs, actions, log_probs, rewards, dones, values = zip(*self.memory)

        advantages = []
        returns = []
        gae = 0
        values = list(values) + [0.0]  # Bootstrap

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        # Convert everything to tensors
        return {
            "obs": torch.FloatTensor(np.array(obs)),
            "actions": torch.FloatTensor(np.array(actions)),
            "log_probs": torch.FloatTensor(log_probs),
            "returns": torch.FloatTensor(returns),
            "advantages": torch.FloatTensor(advantages)
        }

    def update(self):
        batch = self.compute_returns_and_advantages()

        advantages = (batch['advantages'] - batch['advantages'].mean()) / (batch['advantages'].std() + 1e-8)

        for _ in range(self.epochs):
            logits = self.actor(batch['obs'])
            dist = torch.distributions.Normal(logits, 1.0)
            new_log_probs = dist.log_prob(batch['actions']).sum(dim=-1)

            ratio = torch.exp(new_log_probs - batch['log_probs'])
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            value_preds = self.critic(batch['obs']).squeeze(-1)
            value_loss = (batch['returns'] - value_preds).pow(2).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        print(f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")

        self.clear_memory()
