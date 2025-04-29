# scripts/train_ppo.py
import os
import sys
import torch
import numpy as np
import pygame
import matplotlib.pyplot as plt

from agents.ppo_agent import PPOAgent
from environments.simple_env import simpleAUVEnv

# --- CONFIG ---
MAX_STEPS   = 1_000_000
BATCH_SIZE  = 2048
SAVE_MODEL  = "models/ppo_actor_critic.pth"
PLOT_FILE   = "training_plot.png"
WARMUP_EPS  = 40     # ignore first 40 episodes as warm‐up
AVG_WINDOW  = 20     # average over 20‐episode chunks
OUTLIER_STD = 10.0    # drop any episode outside mean±3*std

def save_plot(rewards):
    """
    Compute averages (skipping the first WARMUP_EPS), filter out
    extreme outliers (any r outside mean±OUTLIER_STD*std), and
    save a plot of the rolling‐window means.
    """
    # drop warm‐up
    r = np.array(rewards[WARMUP_EPS:])
    if len(r) < AVG_WINDOW:
        print("Not enough data to plot yet.")
        return

    # compute global mean/std for clipping
    mu, sigma = r.mean(), r.std()
    lo, hi = mu - OUTLIER_STD * sigma, mu + OUTLIER_STD * sigma

    n_chunks = len(r) // AVG_WINDOW
    xs, ys = [], []
    for i in range(n_chunks):
        chunk = r[i*AVG_WINDOW:(i+1)*AVG_WINDOW]
        # filter out extreme outliers
        filt = chunk[(chunk >= lo) & (chunk <= hi)]
        if len(filt) == 0:
            filt = chunk  # if all dropped, fallback to raw
        xs.append(WARMUP_EPS + i*AVG_WINDOW + AVG_WINDOW//2)
        ys.append(np.mean(filt))

    plt.figure()
    plt.plot(xs, ys, marker='o')
    plt.xlabel("Episode")
    plt.ylabel(f"Avg Reward per {AVG_WINDOW} eps (clipped)")
    plt.title("Training Progress (outliers removed)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()
    print(f"→ Saved filtered plot to {PLOT_FILE}")

def main():
    pygame.init()
    # tiny window just to capture keys
    screen = pygame.display.set_mode((200, 50))
    pygame.display.set_caption("Press S to save, Q to quit")

    env = simpleAUVEnv(
        docks=5,
        dock_radius=0.2,
        dock_reward=1000,
        beacon_params={
            'ping_interval':    1.0,
            'pulse_duration':   0.1,
            'beacon_intensity': 1.0,
            'ping_noise':       0.01
        }
    )

    obs = env.reset()
    obs_dim = obs.shape[0]
    act_dim = 2

    agent = PPOAgent(
        obs_dim, act_dim,
        lr=3e-4, gamma=0.99,
        lam=0.95, clip_ratio=0.2,
        epochs=10
    )

    total_steps    = 0
    episode_reward = 0
    episode_steps  = 0
    episode_count  = 0
    all_rewards    = []

    while total_steps < MAX_STEPS:
        # collect one batch
        for _ in range(BATCH_SIZE):
            # handle keypresses
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_s:
                        agent.save(SAVE_MODEL)
                        print(f"→ Model saved to {SAVE_MODEL}")
                        save_plot(all_rewards)
                    elif e.key == pygame.K_q:
                        print("→ Quit requested. Exiting.")
                        pygame.quit()
                        sys.exit()

            action, logp, val = agent.select_action(obs)
            action = np.clip(action, -5.0, 5.0)

            next_obs, rew, done, _ = env.step(action)
            if next_obs is None:
                next_obs = env.reset()

            agent.store_transition((obs, action, logp, rew, done, val))

            obs            = next_obs
            episode_reward += rew
            episode_steps  += 1
            total_steps    += 1

            if done:
                print(f"[Episode {episode_count}] Total Reward: {episode_reward:.3f}, Steps: {episode_steps}")
                all_rewards.append(episode_reward)
                obs = env.reset()
                episode_reward = 0
                episode_steps  = 0
                episode_count += 1

            if total_steps >= MAX_STEPS:
                break

        agent.update()

    # final save & plot when done
    agent.save(SAVE_MODEL)
    print(f"Training finished — model saved to {SAVE_MODEL}")
    save_plot(all_rewards)
    pygame.quit()

if __name__ == "__main__":
    main()
