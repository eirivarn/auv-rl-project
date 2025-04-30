import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from matplotlib import patches, animation
import cv2
import pygame
from tqdm import trange

def train_dqn(env, agent, episodes=500, max_steps=200):
    rewards_hist = []
    pbar = tqdm(range(episodes), desc="DQN Training")
    for ep in pbar:
        state = env.reset()
        total_reward = 0
        done = False

        for t in range(max_steps):
            idx = agent.select_action(state)
            # pass the index directly; step() will handle discrete_actions
            next_s, reward, done, _ = env.step(idx)
            agent.store_transition(state, idx, reward, next_s, done)
            agent.optimize_model()
            state = next_s
            total_reward += reward
            if done:
                break

        agent.update_epsilon()
        rewards_hist.append(total_reward)
        pbar.set_postfix({"Rew": f"{total_reward:.2f}", "ε": f"{agent.epsilon:.3f}"})

    return rewards_hist

def evaluate_agent(env, agent, episodes=100, max_steps=200):
    agent.epsilon = 0.0
    successes, steps = 0, []
    for _ in range(episodes):
        state = env.reset()
        done, t = False, 0
        while not done and t < max_steps:
            idx = agent.select_action(state)
            # pass index for discrete; step() will map it internally
            state, _, done, _ = env.step(idx)
            t += 1
        if done:
            successes += 1
            steps.append(t)
    return successes/episodes, np.mean(steps) if steps else None

def plot_rewards(rewards, window=10):
    episodes = np.arange(len(rewards))
    plt.figure()
    plt.plot(episodes, rewards, alpha=0.3, label="raw")
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(rewards)), ma, label=f"{window}-step MA")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


import cv2
import numpy as np
import pygame
import sys
from tqdm import trange

def record_pygame_robust(env, agent, out_path='auv.avi', max_steps=200, fps=30):
    """
    Robustly record a pygame‐based run of `env` under `agent` to a video file.
    Handles end-of-episode cleanly by breaking before rendering/capture.
    """
    # 1) Initialize Pygame once
    pygame.init()
    width, height = env.window_size
    _ = pygame.display.set_mode((width, height))

    # 2) Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        pygame.quit()
        raise RuntimeError(f"Cannot open video writer for {out_path}")

    try:
        agent.epsilon = 0.0
        obs = env.reset()
        done = False

        for t in trange(max_steps, desc="Recording"):
            # 3) Step agent first, so we don't render after done
            idx = agent.select_action(obs)
            obs, _, done, _ = env.step(idx)
            if done:
                break

            # 4) Handle window events
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    done = True
                    break

            # 5) Render and capture
            env.render()
            surf = pygame.display.get_surface()
            if surf is None:
                break
            arr = pygame.surfarray.array3d(surf)  # (w,h,3)
            frame = cv2.cvtColor(np.transpose(arr, (1,0,2)), cv2.COLOR_RGB2BGR)
            writer.write(frame)

            # 6) Wait to target FPS
            pygame.time.wait(int(1000/fps))

    finally:
        writer.release()
        pygame.quit()

    print(f"Recording saved to {out_path}")

def record_headless(env, agent, out_path='auv.gif', max_steps=200, fps=10):
    """
    Simpler: record frames via env.render(mode='rgb_array') and save a GIF.
    """
    import imageio
    frames = []
    agent.epsilon = 0.0
    obs = env.reset()
    done = False
    t = 0
    while not done and t < max_steps:
        # render offscreen
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        idx = agent.select_action(obs)
        obs, _, done, _ = env.step(idx)
        t += 1

    # write GIF
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Headless recording saved to {out_path}")

# Example usage:
# record_pygame_robust(env, agent, out_path='auv.avi')
# or
# record_headless(env, agent, out_path='auv.gif')
