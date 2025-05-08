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

        # Q-table dimensions: [dx_range x dy_range x actions] = [n_states, n_actions]
        # Minimum and maximum values for the observation space
        minimum_values = env.observation_space.low[0]
        maximum_values = env.observation_space.high[0]
        dx_size = maximum_values - minimum_values + 1
        dy_size = maximum_values - minimum_values + 1

        # Initialize Q-table with zeros
        self.q_table = np.zeros((dx_size, dy_size, env.action_space.n))
        
        # ---- Hyperparameters ----
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rewards_history = []

    def state_to_index(self, obs) -> tuple:
        # shift by low to index into table
        dx_idx = obs[0] - self.env.observation_space.low[0]
        dy_idx = obs[1] - self.env.observation_space.low[1]
        return dx_idx, dy_idx

    def select_action(self, obs) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        dx, dy = self.state_to_index(obs)
        return int(np.argmax(self.q_table[dx, dy]))

    def update(self, 
               state: np.ndarray, 
               action: int, 
               reward: float, 
               next_state: np.ndarray, 
               done) -> None:

        i, j = self.state_to_index(state)
        i_next, j_next = self.state_to_index(next_state)

        current_q_predict = self.q_table[i, j, action]

        future_best_q = np.max(self.q_table[i_next, j_next])

        q_target = reward + (0 if done 
                             else self.gamma * future_best_q)

        td_error = q_target - current_q_predict

        self.q_table[i, j, action] += self.alpha * td_error


    def train(self, 
            episodes=DEFAULT_TRAINING_EPISODES, 
            max_steps=DEFAULT_TRAINING_MAX_STEPS
            ) -> list:
        
        episode_returns = []
        for ep in trange(episodes, desc='Training'):

            reset_output = self.env.reset()

            current_state, _ = (reset_output if isinstance(reset_output, tuple) 
                                                else reset_output)

            total_return = 0
            
            for step in range(max_steps):
                action = self.select_action(current_state)
                next_state, reward, done, _ = self.env.step(action)

                self.update(current_state, action, reward, next_state, done)

                current_state = next_state
                total_return += reward
                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            episode_returns.append(total_return)
        return episode_returns

    def evaluate(self, episodes=100, max_steps=100, render=False) -> float:
        successes = 0
        for ep in range(episodes):
            obs, _ = self.env.reset()
            for step in range(max_steps):
                action = np.argmax(self.q_table[self.state_to_index(obs)])
                obs, _, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                if done:
                    successes += 1
                    break
        success_rate = successes / episodes
        return success_rate

    def save(self, filepath) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath: str):
            """Load Q-table from a .npy file (allow pickled data)."""
            self.q_table = np.load(filepath, allow_pickle=True)
