import gym
from gym import spaces
import numpy as np
import pygame

class GridDockEnv(gym.Env):
    """
    Grid world environment with an autonomous AUV (agent)
    and docking station (goal). State is relative position (dx, dy).
    Actions: 0=up, 1=down, 2=left, 3=right.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=(5, 5), spawn_mode='static', agent_start=None, goal_start=None, cell_size=64, fps=1):
        super(GridDockEnv, self).__init__()
        self.grid_size = grid_size
        assert spawn_mode in ['static', 'random'], "spawn_mode must be 'static' or 'random'"
        self.spawn_mode = spawn_mode

        # Static start positions if provided, else corners
        self.static_agent_start = np.array(agent_start if agent_start is not None else (0, 0), dtype=int)
        self.static_goal_start = np.array(goal_start if goal_start is not None else (grid_size[0] - 1, grid_size[1] - 1), dtype=int)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)

        # Observation is (dx, dy) relative difference
        low = np.array([-grid_size[0] + 1, -grid_size[1] + 1], dtype=int)
        high = np.array([grid_size[0] - 1, grid_size[1] - 1], dtype=int)
        self.observation_space = spaces.Box(low=low, high=high, dtype=int)

        # Internal state
        self.agent_pos = None
        self.goal_pos = None

        # New attributes for render()
        self.grid_size = tuple(grid_size)
        self.spawn_mode = spawn_mode
        self.cell_size  = cell_size
        self._fps       = fps
        self.window_size = (self.grid_size[0] * cell_size,
                            self.grid_size[1] * cell_size)
        self._initialized = False  # <— add this
        self.screen       = None
        self._window      = None
        self.clock        = None

    def reset(self):
        # Initialize positions
        if self.spawn_mode == 'static':
            self.agent_pos = self.static_agent_start.copy()
            self.goal_pos = self.static_goal_start.copy()
        else:
            self.agent_pos = np.random.randint(low=0, high=self.grid_size, size=2)
            self.goal_pos = np.random.randint(low=0, high=self.grid_size, size=2)
        return self._get_obs()

    def step(self, action):
        # Move agent
        if action == 0 and self.agent_pos[1] < self.grid_size[1] - 1:  # up
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0:  # down
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0:  # left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size[0] - 1:  # right
            self.agent_pos[0] += 1

        obs = self._get_obs()
        done = np.array_equal(self.agent_pos, self.goal_pos)
        if done:
            reward = +10.0
        else:
            reward = -0.1
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        # Relative position observation
        return self.goal_pos - self.agent_pos

    def render(self, mode='human'):
        if not self._initialized:
            pygame.init()
            # Must enable the SDL video driver if you plan to do array renders
            # But usually you can skip that in modern jupyter kernels
            self.screen = pygame.Surface(self.window_size)
            # Also keep a window around for human mode:
            self._window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("GridDockEnv")
            self.clock = pygame.time.Clock()
            self._initialized = True

        # Draw background, grid, goal & agent onto self.screen:
        self.screen.fill((255,255,255))
        w, h = self.window_size
        for x in range(0, w, self.cell_size):
            pygame.draw.line(self.screen, (200,200,200), (x,0), (x,h))
        for y in range(0, h, self.cell_size):
            pygame.draw.line(self.screen, (200,200,200), (0,y), (w,y))

        gx, gy = self.goal_pos
        ax, ay = self.agent_pos
        px_goal = gx * self.cell_size
        py_goal = (self.grid_size[1]-1 - gy) * self.cell_size
        goal_rect = pygame.Rect(px_goal, py_goal,
                                self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0,200,0), goal_rect)

        px_agent = ax * self.cell_size
        py_agent = (self.grid_size[1]-1 - ay) * self.cell_size
        cx = px_agent + self.cell_size//2
        cy = py_agent + self.cell_size//2
        pygame.draw.circle(self.screen, (0,0,200),
                           (cx, cy), self.cell_size//3)

        if mode == 'human':
            # Blit the off-screen surface to the window, then flip
            self._window.blit(self.screen, (0,0))
            pygame.display.flip()
            self.clock.tick(self._fps)
        elif mode == 'rgb_array':
            # Return a HxWx3 numpy array (flip X/Y axis from pygame)
            arr = pygame.surfarray.array3d(self.screen)
            return np.transpose(arr, (1,0,2))

    def close(self):
        # Cleanup if needed
        pass
