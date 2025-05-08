# environments/grid_env.py

import numpy as np
import pygame
from collections import deque
import gym
from gym import spaces

from utils.constants import (
    WHITE,
    GRID_COLOR,
    GOAL_COLOR,
    AGENT_COLOR,
    OBSTACLE_COLOR,
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
        self._history_buffer = deque(maxlen=history_length + 1)

        # obstacle config
        if obstacle_positions is not None:
            self.static_obstacles = set(obstacle_positions)
            self.obstacle_count   = 0
        else:
            self.static_obstacles = set()
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

        # action/observation 

        self.action_space = spaces.Discrete(4)

        # observation: [dx,dy] plus optional lidar → integer box
        low  = np.array([-grid_size[0]+1, -grid_size[1]+1], dtype=np.int32)
        high = np.array([ grid_size[0]-1,  grid_size[1]-1], dtype=np.int32)

        if self.use_lidar:
            # extend low/high by 4 more dims (range 0..lidar_range)
            low  = np.concatenate([low, np.zeros(4, dtype=np.int32)])
            high = np.concatenate([high, np.full(4, self.lidar_range, dtype=np.int32)])
        if self.use_history:
            reps = self.history_length + 1
            low  = np.tile(low,  reps)
            high = np.tile(high, reps)

        self.observation_space = spaces.Box(low=low,
                                            high=high,
                                            dtype=np.int32)


    def _compute_lidar(self) -> list:
        """
        Returns a 4-tuple: (d_up, d_down, d_left, d_right)
        Each is # of free cells until obstacle or wall, capped at lidar_range.
        """
        x0, y0 = self.agent_position
        W, H   = self.grid_size
        ranges = []
        # define scanning functions
        directions = [( 0, 1),  # up
                      ( 0,-1),  # down
                      (-1, 0),  # left
                      ( 1, 0)]  # right
        for dx, dy in directions:
            dist = 0
            x, y = x0, y0
            # step until obstacle, wall, or max range
            while dist < self.lidar_range:
                x += dx; y += dy
                if not (0 <= x < W and 0 <= y < H):
                    break
                if (x, y) in self.obstacles:
                    break
                dist += 1
            ranges.append(dist)
        return ranges

    def reset(self) -> tuple:
        # 1) spawn agent & goal
        if self.spawn_mode == 'static':
            self.agent_position = self.static_agent_start.copy()
            self.goal_position  = self.static_goal_start.copy()
        else:
            self.agent_position = np.random.randint(0, self.grid_size, size=2)
            self.goal_position  = np.random.randint(0, self.grid_size, size=2)

        # 2) set up obstacles
        if self.static_obstacles:
            self.obstacles = set(self.static_obstacles)
        elif self.obstacle_count > 0:

            all_cells = {(x,y) for x in range(self.grid_size[0])
                               for y in range(self.grid_size[1])}
            forbidden = {tuple(self.agent_position), tuple(self.goal_position)}
            choices   = list(all_cells - forbidden)
            np.random.shuffle(choices)
            self.obstacles = set(choices[:self.obstacle_count])
        else:
            self.obstacles = set()
        
        # 3) initialize history buffer
        if self.use_history:
            obs = self._get_raw_obs()
            self._history_buffer.clear()
            for _ in range(self.history_length+1):
                self._history_buffer.append(obs.copy())

        return self._get_obs(), {}

    def step(self, action) -> tuple:
        """
        Executes the action, updates agent_position, obstacles, reward, done,
        and maintains the history buffer if use_history=True.
        """
        # 1) Move agent
        x, y = self.agent_position
        if   action == 0: y += 1   # up
        elif action == 1: y -= 1   # down
        elif action == 2: x -= 1   # left
        elif action == 3: x += 1   # right

        # 2) Bounds check
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            x, y = self.agent_position

        # 3) Obstacle check
        if (x, y) in self.obstacles:
            x, y = self.agent_position

        self.agent_position = np.array((x, y), dtype=int)

        # 4) Compute reward & done
        done = np.array_equal(self.agent_position, self.goal_position)
        reward = 1.0 if done else -1.0

        # 5) Get the new “raw” obs (dx,dy + optional LiDAR)
        raw_obs = self._get_raw_obs()

        # 6) Update history buffer (if enabled)
        if self.use_history:
            # append latest raw_obs; deque will drop oldest if full
            self._history_buffer.append(raw_obs.copy())
            # build the stacked observation
            obs = np.concatenate(list(self._history_buffer), axis=0)
        else:
            obs = raw_obs

        return obs, reward, done, {}

    def _get_raw_obs(self) -> np.ndarray:
        """ Returns the basic relative vector (dx,dy) or with LiDAR appended. """
        basic = list(self.goal_position - self.agent_position)
        if self.use_lidar:
            basic += self._compute_lidar()
        return np.array(basic, dtype=int)

    def _get_obs(self) -> np.ndarray:
        if not self.use_history:
            return self._get_raw_obs()
        # otherwise flatten the history buffer
        flat = []
        for past in self._history_buffer:
            flat.extend(past.tolist())
        return np.array(flat, dtype=int)

    def render(self, mode='human') -> None:
        if not self._initialized:
            pygame.init()
            # off-screen surface
            self.screen = pygame.Surface(self.window_size)
            # visible window
            self._window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("GridDockEnv")
            self.clock = pygame.time.Clock()
            self._initialized = True

        # draw background
        self.screen.fill((255,255,255))

        W, H = self.grid_size
        # grid lines
        for x in range(W+1):
            pygame.draw.line(self.screen, (200,200,200),
                             (x*self.cell_size, 0),
                             (x*self.cell_size, H*self.cell_size))
        for y in range(H+1):
            pygame.draw.line(self.screen, (200,200,200),
                             (0, y*self.cell_size),
                             (W*self.cell_size, y*self.cell_size))

        # draw obstacles as black squares
        for (ox,oy) in self.obstacles:
            rect = pygame.Rect(
                ox*self.cell_size,
                (H-1-oy)*self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, (0,0,0), rect)

        # draw goal (green)
        gx, gy = self.goal_position
        goal_rect = pygame.Rect(
            gx*self.cell_size,
            (H-1-gy)*self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, (0,200,0), goal_rect)

        # draw agent (blue circle)
        ax, ay = self.agent_position
        cx = ax*self.cell_size + self.cell_size//2
        cy = (H-1-ay)*self.cell_size + self.cell_size//2
        pygame.draw.circle(self.screen, (0,0,200), (cx,cy), self.cell_size//3)

        # blit & flip
        self._window.blit(self.screen, (0,0))
        pygame.display.flip()
        if mode=='human':
            self.clock.tick(self._fps)

    def close(self) -> None:
        if self._initialized:
            pygame.quit()
            self._initialized = False
