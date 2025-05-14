# environments/grid_env.py

import numpy as np
import pygame
from collections import deque
import gym
from gym import spaces

from environments.grid_env.grid_utils import place_obstacles, move, Lidar
from utils.shared_utils import HistoryBuffer

from environments.grid_env.grid_constants import (
    DEFAULT_GRID_SIZE,
    DEFAULT_CELL_SIZE,
    DEFAULT_FPS,
    DEFAULT_LIDAR_RANGE,
    DEFAULT_HISTORY_LENGTH,
    DEFAULT_SPAWN_MODE,
    DEFAULT_USE_LIDAR,
    DEFAULT_USE_HISTORY,
    DEFAULT_OBSTACLE_COUNT,
    DEFAULT_OBSTACLE_POSITIONS,
    DEFAULT_FPS,
)

class GridEnv(gym.Env):
    def __init__(self,
                 grid_size=DEFAULT_GRID_SIZE,
                 spawn_mode=DEFAULT_SPAWN_MODE,
                 cell_size=DEFAULT_CELL_SIZE,
                 fps=DEFAULT_FPS,
                 obstacle_positions=DEFAULT_OBSTACLE_POSITIONS,
                 obstacle_count= DEFAULT_OBSTACLE_COUNT,
                 use_lidar: bool = DEFAULT_USE_LIDAR,
                 lidar_range: int = DEFAULT_LIDAR_RANGE,
                 use_history: bool = DEFAULT_USE_HISTORY,
                 history_length: int = DEFAULT_HISTORY_LENGTH
                 ):

        assert spawn_mode in ('static','random')
        self.grid_size = tuple(grid_size)
        self.spawn_mode = spawn_mode
        self.cell_size  = cell_size
        self._fps       = fps
        self.use_lidar   = use_lidar
        self.lidar_range = lidar_range
        self.use_history     = use_history
        self.history_length  = history_length

        # obstacle config
        if obstacle_positions is not None:
            self.static_obstacles = set(obstacle_positions)
            self.obstacle_count   = 0
        else:
            self.static_obstacles = None
            self.obstacle_count   = int(obstacle_count)

        # fixed agent & goal for static mode
        self.static_agent_start = np.array((0, 0), dtype=int)
        self.static_goal_start  = np.array((grid_size[0]-1, grid_size[1]-1), dtype=int)

        # placeholders
        self.agent_position = None
        self.goal_position  = None
        self.obstacles = set()

        # rendering internals
        self.window_size  = (grid_size[0]*cell_size, grid_size[1]*cell_size)
        self._initialized = False
        self.screen       = None
        self._window      = None
        self.clock        = None

        self.action_space = spaces.Discrete(4)

        low  = np.array([-grid_size[0]+1, -grid_size[1]+1], dtype=np.int32)
        high = np.array([ grid_size[0]-1,  grid_size[1]-1], dtype=np.int32)

        if self.use_lidar:
            low  = np.concatenate([low, np.zeros(4, dtype=np.int32)])
            high = np.concatenate([high, np.full(4, self.lidar_range, dtype=np.int32)])
        if self.use_history:
            reps = self.history_length + 1
            low  = np.tile(low,  reps)
            high = np.tile(high, reps)

        self.sensors = []
        if self.use_lidar:
            self.sensors.append(Lidar(self, lidar_range))
        if self.use_history:
            self.sensors.append(HistoryBuffer(history_length))

        self.observation_space = spaces.Box(low=low,
                                            high=high,
                                            dtype=np.int32)

    def reset(self) -> tuple:

        if self.spawn_mode == 'static':
            self.agent_position = self.static_agent_start.copy()
            self.goal_position  = self.static_goal_start.copy()
        else:
            self.agent_position = np.random.randint(0, self.grid_size, size=2)
            self.goal_position  = np.random.randint(0, self.grid_size, size=2)

        self.obstacles = place_obstacles(self.agent_position,
                                         self.goal_position,
                                         self.grid_size,
                                         static_positions=self.static_obstacles,
                                         count=self.obstacle_count)
        
        raw_observations = np.array(self.goal_position - self.agent_position, dtype=int)

        observation = raw_observations
        for sensor in self.sensors:
            observation = sensor.reset(observation)

        if self.observation_space is None:
            observation_shape = observation.shape[0]
            low = -np.array(self.grid_size)+1
            high = np.array(self.grid_size)-1
            self.observation_space = spaces.Box(low=np.full(observation_shape, low.min(),int),
                                                high=np.full(observation_shape, high.max(),int),
                                                dtype=np.int32)
        return observation, {}
    
    def step(self, action) -> tuple:
        self.agent_position = move(self.agent_position, action, self.grid_size, self.obstacles)

        done = np.array_equal(self.agent_position, self.goal_position)
        reward = 1.0 if done else -1.0

        raw_observation = np.array(self.goal_position - self.agent_position, dtype=int)
        observation = raw_observation
        for sensor in self.sensors:
            observation = sensor.process(observation)

        return observation, reward, done, {}
