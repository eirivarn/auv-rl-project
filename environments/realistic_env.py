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
        use_continuous_reward: bool = False,
        step_penalty: float       = -1.0,
        slow_step_penalty: float  = -0.01,
        action_penalty_coeff:float= 0.1,
        progress_coeff: float      = 5.0, 
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
            self.cur_strength    = current_params.get('strength', 0.0)
            self.cur_period      = current_params.get('period', 30.0)
            self.cur_direction   = current_params.get('direction', 0.0)
        else:
            self.current_enabled = False

        # action setup: discrete strafing or continuous
        self.turn_penalty_coeff = 0.5
        if self.discrete_actions:
            self.actions = [
                ( 0.5,  0.0,  0.0),  # forward
                (-0.5,  0.0,  0.0),  # back
                ( 0.0,  0.5,  0.0),  # strafe left
                ( 0.0, -0.5,  0.0),  # strafe right
                ( 0.0,  0.0,  0.3),  # pivot left
                ( 0.0,  0.0, -0.3),  # pivot right
            ]
            self.action_space = spaces.Discrete(len(self.actions))
        else:
            # ─── New: Continuous 3-DoF bounds ──────────────────────────
            # forward speed limit (m/s)
            self.v_limit     = 0.05
            # lateral (strafe) speed limit (m/s)
            self.lat_limit   = 0.05
            # yaw rate limit (rad/s)
            self.omega_limit = np.pi / 16

            low = np.array(
                [-self.v_limit, -self.lat_limit, -self.omega_limit],
                dtype=np.float32
            )
            high = np.array(
                [ self.v_limit,  self.lat_limit,  self.omega_limit],
                dtype=np.float32
            )
            self.action_space = spaces.Box(low, high, dtype=np.float32)
            self.actions = None

            # — flag to choose reward regime —
            self.use_continuous_reward = use_continuous_reward
            # — parameters for both schemes —
            self.step_penalty        = step_penalty
            self.slow_step_penalty   = slow_step_penalty
            self.action_penalty_coeff= action_penalty_coeff
            self.progress_coeff      = progress_coeff
            self.collision_penalty   = -10.0

            self._last_dist = None

    def reset(self):
        # 0) re-sample a new dock (preserve the number of docks)
        num = len(self.docks)
        self.docks = [ self._sample_random_goal() for _ in range(num) ]

        # 1) now call the parent reset() to place AUV, point at the new dock, etc.
        obs = super().reset()

        # 2) zero out your velocity/time
        self.vel[:] = 0.0
        self.time   = 0.0

        self._last_dist = np.linalg.norm(self.pose[:2] - self.docks[0])
        return obs


    def step(self, action):
        # ─── Decode & Clip ───────────────────────────────────────────
        if self.discrete_actions:
            # Unchanged discrete handling
            v_cmd, lat_cmd, omega_cmd = self.actions[int(action)]
        else:
            # Accept tuple, list, or ndarray of length 3
            arr = np.asarray(action, dtype=float).flatten()
            if arr.shape != (3,):
                raise ValueError(f"Invalid action shape: {action}; expected length 3")
            v_cmd, lat_cmd, omega_cmd = arr.tolist()

        # Clip to your desired limits (example limits here)
        v_limit     = 0.05             # max forward/back speed
        lat_limit   = 0.05              # max lateral (strafe) speed
        omega_limit = np.pi / 8        # max yaw rate

        v_cmd     = float(np.clip(v_cmd,   -v_limit,   v_limit))
        lat_cmd   = float(np.clip(lat_cmd, -lat_limit, lat_limit))
        omega_cmd = float(np.clip(omega_cmd, -omega_limit, omega_limit))

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
    
        # 1) distance & progress
        d     = np.linalg.norm(self.pose[:2] - self.docks[0])
        delta = self._last_dist - d

        # 2) base reward + done flag
        done = False
        if self.reward_mode == "discrete":
            # exactly your old scheme
            if collided:
                reward = self.collision_penalty
            elif d < self.dock_radius:
                reward, done = self.dock_reward, True
            else:
                reward = self.step_penalty

            # shaping
            reward += self.progress_coeff * delta
            reward -= self.turn_penalty_coeff * abs(omega_cmd)

        else:  # continuous‐shaped rewards
            # small constant penalty per step
            reward = -self.slow_step_penalty 

            # penalize large, jerky commands
            reward -= self.action_penalty_coeff * (
                abs(v_cmd) + abs(lat_cmd) + abs(omega_cmd)
            )

            # reward for moving closer
            reward += self.progress_coeff * delta

            # collision penalty still applies
            if collided:
                reward += self.collision_penalty

            # final docking bonus
            if d < self.dock_radius:
                reward += self.dock_reward
                done = True

        # 3) optional wall‐proximity penalty (same in both modes)
        raw_ranges, _, _ = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )
        min_r = raw_ranges.min()
        if min_r < self.wall_thresh:
            reward -= self.wall_penalty_coeff * (1 - min_r/self.wall_thresh)

        # 4) record last distance and return
        self._last_dist = d

        # build obs
        raw = self._get_raw_obs()
        if self.use_history:
            self._history_buffer.append(raw.copy())
            obs = np.concatenate(self._history_buffer, axis=0)
        else:
            obs = raw

        return obs, reward, done, {}
