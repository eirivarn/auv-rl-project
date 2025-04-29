import torch
import pygame
import numpy as np
from agents.ppo_agent import PPOAgent
from environments.simple_env import simpleAUVEnv

def main():
    pygame.init()

    # Create env exactly as during training
    env = simpleAUVEnv(
        docks=3,
        dock_radius=0.5,
        dock_reward=50,
        beacon_params={
            'ping_interval':    1.0,
            'pulse_duration':   0.1,
            'beacon_intensity': 1.0,
            'ping_noise':       0.01
        }
    )

    # Grab obs_dim automatically
    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = 2  # (v, omega)

    # Load your trained agent
    agent = PPOAgent(obs_dim, act_dim)
    agent.load("models/ppo_actor_critic.pth")

    # Run one episode at fixed frame-rate
    obs    = env.reset()
    clock  = pygame.time.Clock()
    done   = False

    while not done:
        # deterministic=True => use the network mean (no sampling noise)
        action, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        env.render()
        clock.tick(10)  # 10 FPS for human‐viewable playback

    pygame.quit()

if __name__ == "__main__":
    main()
