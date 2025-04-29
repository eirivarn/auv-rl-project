import torch
import numpy as np
import pygame
from agents.ppo_agent import PPOAgent
from environments.simple_env import simpleAUVEnv

def main():
    pygame.init()

    env = simpleAUVEnv()

    # Updated observation dimension:
    # n_beams (ranges) + n_beams (intensities if enabled) + 6 scalar features
    n_beams = env.sonar.n_beams
    include_intensity = env.sonar.compute_intensity
    obs = env.reset()
    print("Obs shape:", obs.shape)
    obs_dim = obs.shape[0]
    

    # Still (v, omega) → 2 continuous actions
    act_dim = 2

    agent = PPOAgent(obs_dim, act_dim)

    max_steps = 50_000
    batch_size = 2048
    total_steps = 0

    obs, _, done, _ = env.reset(), 0, False, {}
    print("Obs shape:", obs.shape)  # ← Add this

    episode_reward = 0
    episode_steps = 0
    episode_count = 0

    while total_steps < max_steps:
        step = 0
        while step < batch_size:
            action, log_prob, value = agent.select_action(obs)

            # Clip actions to safe values if necessary
            action = np.clip(action, -5.0, 5.0)

            next_obs, reward, done, _ = env.step(action)

            if next_obs is None:
                next_obs = env.reset()

            agent.store_transition((obs, action, log_prob, reward, done, value))

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            step += 1
            total_steps += 1

            if done:
                print(f"[Episode {episode_count}] Total Reward: {episode_reward:.3f}, Steps: {episode_steps}")
                obs, _, done, _ = env.reset(), 0, False, {}
                episode_reward = 0
                episode_steps = 0
                episode_count += 1

        agent.update()

    print("Training finished.")
    pygame.quit()

if __name__ == "__main__":
    main()
