import numpy as np
import math
from environments.simple_env import simpleAUVEnv
from gym import spaces

class realisticAUVEnv(simpleAUVEnv):
    """
    Extends simpleAUVEnv by adding simple Newtonian dynamics, drag, and ocean currents,
    and supports both discrete strafing actions and continuous 3-DoF thrust.
    """
    def __init__(
        self,
        mass: float = 1.0,
        drag_coef: float = 0.1,
        current_params: dict = None,
        dt: float = 0.1,
        **kwargs
    ):
        # predeclare dynamics state
        self.vel  = np.zeros(2, dtype=float)
        self.time = 0.0

        # initialize base env (builds maps, sets pose, calls reset)
        super().__init__(**kwargs)

        # physics parameters
        self.mass      = mass
        self.drag_coef = drag_coef
        self.dt        = dt

        # ocean currents
        if current_params is not None:
            self.current_enabled = True
            self.cur_strength    = current_params.get('strength', 0.1)
            self.cur_period      = current_params.get('period', 30.0)
            self.cur_direction   = current_params.get('direction', 0.0)
        else:
            self.current_enabled = False

        # action setup: discrete strafing or continuous
        self.turn_penalty_coeff = 0.5
        if self.discrete_actions:
            # six discrete maneuvers: forward, back, strafe L/R, yaw L/R
            self.actions = [
                ( 0.5,  0.0,  0.0),  # forward
                (-0.5,  0.0,  0.0),  # back
                ( 0.0,  0.5,  0.0),  # strafe left
                ( 0.0, -0.5,  0.0),  # strafe right
                ( 0.0,  0.0,  0.3),  # pivot left
                ( 0.0,  0.0, -0.3),  # pivot right
            ]
                # ( 0.0,  0.0,  0.3),  # pivot left
                # ( 0.0,  0.0, -0.3),  # pivot right
            self.action_space = spaces.Discrete(len(self.actions))
        else:
            # continuous 3-DoF: (forward, lateral, yaw)
            self.actions = None
            low  = np.array([-1.0, -1.0, -np.pi/4], dtype=np.float32)
            high = np.array([ 1.0,  1.0,  np.pi/4], dtype=np.float32)
            self.action_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        # 0) re-sample a new dock (preserve the number of docks)
        num = len(self.docks)
        self.docks = [ self._sample_random_goal() for _ in range(num) ]

        # 1) now call the parent reset() to place AUV, point at the new dock, etc.
        obs = super().reset()

        # 2) zero out your velocity/time
        self.vel[:] = 0.0
        self.time   = 0.0

        return obs


    def step(self, action):
        # decode action
        if self.discrete_actions:
            v_cmd, lat_cmd, omega_cmd = self.actions[int(action)]
        else:
            if isinstance(action, tuple) and len(action) == 3:
                v_cmd, lat_cmd, omega_cmd = action
            elif isinstance(action, tuple) and len(action) == 2:
                v_cmd, omega_cmd = action
                lat_cmd = 0.0
            else:
                raise ValueError(f"Invalid action shape: {action}")
        # clip
        v_cmd     = float(np.clip(v_cmd,     -1.0,  1.0))
        lat_cmd   = float(np.clip(lat_cmd,   -1.0,  1.0))
        omega_cmd = float(np.clip(omega_cmd, -np.pi/4, np.pi/4))

        old_x, old_y, old_th = self.pose

        # compute current
        if self.current_enabled:
            mag     = self.cur_strength * math.sin(2*math.pi * self.time / self.cur_period)
            current = mag * np.array([math.cos(self.cur_direction), math.sin(self.cur_direction)])
        else:
            current = np.zeros(2)

        # thrust and drag
        F_body  = np.array([v_cmd, lat_cmd])
        R       = np.array([[math.cos(old_th), -math.sin(old_th)], [math.sin(old_th), math.cos(old_th)]])
        F_thrust = R.dot(F_body)
        rel_v   = self.vel - current
        F_drag  = -self.drag_coef * rel_v

        # integrate dynamics
        acc      = (F_thrust + F_drag) / self.mass
        self.vel += acc * self.dt
        new_x    = old_x + self.vel[0] * self.dt
        new_y    = old_y + self.vel[1] * self.dt
        new_th   = math.atan2(math.sin(old_th + omega_cmd * self.dt), math.cos(old_th + omega_cmd * self.dt))

        # collision & bounds check
        dx, dy = new_x - old_x, new_y - old_y
        steps   = max(1, int(math.hypot(dx, dy) / (self.resolution * 0.3)))
        collided = False
        for i in range(1, steps+1):
            xi = old_x + dx * (i/steps)
            yi = old_y + dy * (i/steps)
            ci = int(np.clip(xi/self.resolution, 0, self.grid_size[1]-1))
            ri = int(np.clip(yi/self.resolution, 0, self.grid_size[0]-1))
            if self.occ_grid[ri, ci]:
                collided = True
                break

        oob = not (0 <= new_x <= self.grid_size[1]*self.resolution and 0 <= new_y <= self.grid_size[0]*self.resolution)

        # collision penalty
        reward = self.collision_penalty if collided else 0.0

        # committing pose: block through walls
        if collided or oob:
            # do not move, but update heading
            self.pose = np.array([old_x, old_y, new_th], dtype=float)
        else:
            self.pose = np.array([new_x, new_y, new_th], dtype=float)

        # advance time
        self.time += self.dt

        # reward shaping & goal
        d = np.linalg.norm(self.pose[:2] - self.docks[0])
        if d < self.dock_radius:
            reward, done = self.dock_reward, True
        else:
            reward += -1.0
            done = False

        raw_ranges, _, _ = self.sonar.get_readings(self.occ_grid, self.refl_grid, self.pose)
        min_r = raw_ranges.min()
        if min_r < self.wall_thresh:
            reward -= self.wall_penalty_coeff * (1 - min_r/self.wall_thresh)

        delta = self._last_dist - d
        reward += delta * self.progress_coeff
        self._last_dist = d
        reward -= self.turn_penalty_coeff * abs(omega_cmd)

        # observation
        raw = self._get_raw_obs()
        if self.use_history:
            self._history_buffer.append(raw.copy())
            obs = np.concatenate(self._history_buffer, axis=0)
        else:
            obs = raw

        return obs, reward, done, {}
