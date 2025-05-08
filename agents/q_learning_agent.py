import numpy as np
import random
import pickle
from tqdm import trange

from utils.constants import (
    DEFAULT_EPSILON_START,
    DEFAULT_EPSILON_END,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_ALPHA,
    DEFAULT_GAMMA,
    DEFAULT_TRAINING_EPISODES,
    DEFAULT_TRAINING_MAX_STEPS
)

class QLearningAgent:
    def __init__(self, 
                 env,
                alpha=DEFAULT_ALPHA, 
                gamma=DEFAULT_GAMMA, 
                epsilon=DEFAULT_EPSILON_START, 
                epsilon_min=DEFAULT_EPSILON_END, 
                epsilon_decay=DEFAULT_EPSILON_DECAY,
                ):
        
        self.env = env
        # Q-table dimensions: dx_range x dy_range x actions
        dx_size = env.observation_space.high[0] - env.observation_space.low[0] + 1
        dy_size = env.observation_space.high[1] - env.observation_space.low[1] + 1
        self.q_table = np.zeros((dx_size, dy_size, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rewards_history = []

    def obs_to_index(self, obs):
        # shift by low to index into table
        dx_idx = obs[0] - self.env.observation_space.low[0]
        dy_idx = obs[1] - self.env.observation_space.low[1]
        return dx_idx, dy_idx

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        dx, dy = self.obs_to_index(obs)
        return int(np.argmax(self.q_table[dx, dy]))

    def update(self, obs, action, reward, next_obs, done):
        dx, dy = self.obs_to_index(obs)
        nx, ny = self.obs_to_index(next_obs)
        q_predict = self.q_table[dx, dy, action]
        q_target = reward + (0 if done else self.gamma * np.max(self.q_table[nx, ny]))
        self.q_table[dx, dy, action] += self.alpha * (q_target - q_predict)

    def train(self, 
            episodes=DEFAULT_TRAINING_EPISODES, 
            max_steps=DEFAULT_TRAINING_MAX_STEPS
            ):
        for ep in trange(episodes, desc='Training'):
            obs, _ = self.env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = self.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward
                if done:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.rewards_history.append(total_reward)
        return self.rewards_history

    def evaluate(self, episodes=100, max_steps=100, render=False):
        successes = 0
        for ep in range(episodes):
            obs, _ = self.env.reset()
            for step in range(max_steps):
                action = np.argmax(self.q_table[self.obs_to_index(obs)])
                obs, _, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                if done:
                    successes += 1
                    break
        success_rate = successes / episodes
        return success_rate

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath: str):
            """Load Q-table from a .npy file (allow pickled data)."""
            self.q_table = np.load(filepath, allow_pickle=True)
