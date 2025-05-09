from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pygame
from tqdm import trange

from utils.constants import DIRECTIONS

def move(position: np.ndarray, action:int, grid_size: tuple, obstacles: set) -> np.ndarray:
    x, y = position
    if action == 0:  # Up
        y += 1
    elif action == 1:  # Down
        y -= 1
    elif action == 2:  # Left
        x -= 1
    elif action == 3:  # Right
        x += 1

    if not (0 <= x < grid_size[0] and 0 <= y < grid_size[1]):
             return position
    if (x, y) in obstacles:
        return position
    return np.array((x, y), dtype=int)

def place_obstacles(agent_position: np.ndarray, goal_position: np.ndarray, grid_size: tuple, static_positions = None, count: int = 0) -> set:
    if static_positions is not None: 
        return static_positions
    
    all_cells = {(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])}
    forbidden_cells = {tuple(agent_position), tuple(goal_position)}
    free_cells = list(all_cells - forbidden_cells)
    np.random.shuffle(free_cells)
    return set(free_cells[:count])

def get_raw_observation(self) -> np.ndarray:
    basic = list(self.goal_position - self.agent_position)
    if self.use_lidar:
        basic += self._compute_lidar()
    return np.array(basic, dtype=int)

def get_observation(self) -> np.ndarray:
    if not self.use_history:
        return self._get_raw_obs()
    # otherwise flatten the history buffer
    flat = []
    for past in self._history_buffer:
        flat.extend(past.tolist())
    return np.array(flat, dtype=int)

def evaluate_agent(env, agent, episodes=100, max_steps=200):
    agent.epsilon = 0.0
    successes, steps = 0, []

    for _ in range(episodes):
        state, _ = env.reset()
        done, t = False, 0
        final_reward = None

        while not done and t < max_steps:
            # select_action returns an int (the action index)
            idx = agent.select_action(state)
            state, reward, done, _ = env.step(idx)
            final_reward = reward
            t += 1

        # only count it if the terminal reward was positive (dock reached)
        if final_reward is not None and final_reward > 0:
            successes += 1
            steps.append(t)

    success_rate = successes / episodes
    avg_steps   = np.mean(steps) if steps else None
    return success_rate, avg_steps


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

class Lidar:
    def __init__(self, env, lidar_range: int):
        self.env = env
        self.lidar_range = lidar_range

    def reset(self, raw_observations: np.ndarray) -> np.ndarray:
        return self.process(raw_observations)
    
    def process(self, raw_observations: np.ndarray) -> np.ndarray:
        lidar_hit_readings = []
        x0, y0 = self.env.agent_position
        W, H = self.env.grid_size

        for dx, dy in DIRECTIONS:
            readings = -0
            x, y = x0, y0
            while readings < self.lidar_range:
                x += dx
                y += dy
                if not (0 <= x < W and 0 <= y < H) or (x, y) in self.env.obstacles:
                    break
                readings += 1
            lidar_hit_readings.append(readings)
        return np.concatenate([raw_observations, np.array(lidar_hit_readings, dtype=int)])
    
class HistoryBuffer:
    def __init__(self, history_length: int):
        self.history_length = history_length
        self.buffer = deque(maxlen=history_length+1)

    def reset(self, raw_observations: np.ndarray) -> np.ndarray:
        self.buffer.clear()
        for _ in range(self.history_length+1):
            self.buffer.append(raw_observations.copy())
        return np.concatenate(self.buffer)
    
    def process(self, raw_observations: np.ndarray) -> np.ndarray:
        self.buffer.append(raw_observations.copy())
        return np.concatenate(self.buffer)


def train_dqn(env, agent, episodes=500, max_steps=200):
    rewards_hist = []
    pbar = tqdm(range(episodes), desc="DQN Training")
    for ep in pbar:
        state, _ = env.reset()
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
        state, _ = env.reset()
        done = False

        for t in trange(max_steps, desc="Recording"):
            # 3) Step agent first, so we don't render after done
            idx = agent.select_action(state)
            state, _, done, _ = env.step(idx)
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
    import imageio
    frames = []
    agent.epsilon = 0.0
    state, _ = env.reset()
    done = False
    t = 0
    while not done and t < max_steps:
        # render offscreen
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        idx = agent.select_action(state)
        obs, _, done, _ = env.step(idx)
        t += 1

    # write GIF
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Headless recording saved to {out_path}")
