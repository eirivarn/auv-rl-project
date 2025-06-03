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

    total_successes = 0
    steps_list = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        t = 0

        while not done and t < max_steps:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            t += 1

        if hasattr(env, 'pose') and hasattr(env, 'docks') and hasattr(env, 'dock_radius'):
            try:
                if np.linalg.norm(env.pose[:2] - env.docks[0]) < env.dock_radius:
                    total_successes += 1
                    steps_list.append(t)
            except Exception:
                if done:
                    total_successes += 1
                    steps_list.append(t)
        else:
            if done:
                total_successes += 1
                steps_list.append(t)

    success_rate = total_successes / episodes
    avg_steps = np.mean(steps_list) if steps_list else None
    return success_rate, avg_steps
