from collections import deque

import numpy as np


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

def evaluate_agent(env, agent, episodes=100, max_steps=200):
    agent.epsilon = 0.0
    successes, steps = 0, []

    for _ in range(episodes):
        state, _ = env.reset()
        done, t = False, 0
        final_reward = None

        while not done and t < max_steps:
            idx = agent.select_action(state)
            state, reward, done, _ = env.step(idx)
            final_reward = reward
            t += 1

        if np.linalg.norm(env.pose[:2] - env.docks[0]) < env.dock_radius:
            successes += 1
            steps.append(t)

    success_rate = successes / episodes
    avg_steps   = np.mean(steps) if steps else None
    return success_rate, avg_steps