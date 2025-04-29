import torch
import torch.nn as nn
import numpy as np
import pygame
from environments.simple_env import simpleAUVEnv

# 1. Define a small Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)

def main():
    pygame.init()
    env = simpleAUVEnv(
        sonar_params={'compute_intensity': False},
        current_params=None,
        goal_params={'radius': 0.5},
        beacon_params={
            'ping_interval': 1.0,
            'pulse_duration': 0.1,
            'beacon_intensity': 1.0,
            'ping_noise': 0.01
        },
        window_size=(800, 600)
    )

    obs = env.reset()
    obs_dim = len(obs)
    act_dim = 2  # (v, omega)

    policy = PolicyNetwork(obs_dim, act_dim)

    episode = 0
    total_reward = 0.0
    steps = 0

    running = True
    clock = pygame.time.Clock()

    while running:
        # --- Select action ---
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Shape: (1, obs_dim)
        action = policy(obs_tensor).squeeze(0).detach().numpy()

        # Optional: Clip the actions to safe values
        action = np.clip(action, -1.0, 1.0)  # v, omega between -1 and 1

        # --- Step environment ---
        obs, reward, done, _ = env.step(action)

        total_reward += reward
        steps += 1

        env.render()
        clock.tick(30)

        if done:
            print(f"[Episode {episode}] Total Reward: {total_reward:.3f}, Steps: {steps}")
            episode += 1
            total_reward = 0.0
            steps = 0
            obs = env.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    main()
