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

