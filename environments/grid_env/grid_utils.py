from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pygame
from tqdm import trange

from environments.grid_env.grid_constants import DIRECTIONS

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
    flat = []
    for past in self._history_buffer:
        flat.extend(past.tolist())
    return np.array(flat, dtype=int)


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
    
