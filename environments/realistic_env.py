import numpy as np
import math
from environments.simple_env import simpleAUVEnv
from gym import spaces

def generate_random_map(grid_size, fill_prob=0.3, smooth_steps=5, birth_limit=6, death_limit=4):
    H, W = grid_size
    grid = (np.random.rand(H, W) < fill_prob).astype(np.uint8)

    def count_walls(y, x):
        total = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    total += grid[ny, nx]
                else:
                    total += 1
        return total

    for _ in range(smooth_steps):
        new_grid = np.zeros_like(grid)
        for y in range(H):
            for x in range(W):
                walls = count_walls(y, x)
                if grid[y, x] == 1:
                    new_grid[y, x] = 1 if walls >= death_limit else 0
                else:
                    new_grid[y, x] = 1 if walls >= birth_limit else 0
        grid = new_grid

    return grid

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
        use_continuous_reward=False,
        use_drag: bool = False,
        use_inertia: bool   = False,
        step_penalty=-1.0,
        slow_step_penalty=0.01,        
        action_penalty_coeff=0.1,
        progress_coeff=5.0,
        collision_penalty=-5.0,
        turn_penalty_coeff=0.05,
        v_limit=0.3,
        lat_limit=0.3,
        omega_limit= np.pi / 16,
        **kwargs
    ):
        self.vel  = np.zeros(2, dtype=float)
        self.time = 0.0

        # Extract map-related kwargs (remove them before calling super())
        random_map     = kwargs.pop("random_map", False)
        map_fill_prob  = kwargs.pop("map_fill_prob", 0.3)
        smooth_steps   = kwargs.pop("smooth_steps", 5)
        birth_limit    = kwargs.pop("birth_limit", 6)
        death_limit    = kwargs.pop("death_limit", 4)
        grid_size      = kwargs.get("grid_size", (200, 200))

        super().__init__(**kwargs)

        if random_map:
            self.occ_grid = generate_random_map(grid_size, map_fill_prob, smooth_steps, birth_limit, death_limit)
            self.refl_grid = np.full(grid_size, 0.2, dtype=np.float32)

        self.use_continuous_reward   = use_continuous_reward
        self.step_penalty            = step_penalty
        self.slow_step_penalty       = slow_step_penalty
        self.action_penalty_coeff    = action_penalty_coeff
        self.progress_coeff          = progress_coeff
        self.collision_penalty       = collision_penalty
        self.turn_penalty_coeff      = turn_penalty_coeff

        self.use_drag  = use_drag
        self.use_inertia = use_inertia
        self.mass      = mass
        self.drag_coef = drag_coef
        self.dt        = dt

        if current_params is not None:
            self.current_enabled = True
            self.cur_strength    = current_params.get('strength', 0.0)
            self.cur_period      = current_params.get('period', 30.0)
            self.cur_direction   = current_params.get('direction', 0.0)
        else:
            self.current_enabled = False

        if self.discrete_actions:
            self.actions = [
                ( 0.5,  0.0,  0.0),
                (-0.5,  0.0,  0.0),
                ( 0.0,  0.5,  0.0),
                ( 0.0, -0.5,  0.0),
                ( 0.0,  0.0,  0.3),
                ( 0.0,  0.0, -0.3),
            ]
            self.action_space = spaces.Discrete(len(self.actions))
        else:
            self.v_limit     = v_limit
            self.lat_limit   = lat_limit
            self.omega_limit = omega_limit

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

         # ─── initialize last-distance for progress shaping ──────────
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

        # Clip to the limits you set in __init__
        v_cmd     = float(np.clip(v_cmd,   -self.v_limit,   self.v_limit))
        lat_cmd   = float(np.clip(lat_cmd, -self.lat_limit, self.lat_limit))
        omega_cmd = float(np.clip(omega_cmd, -self.omega_limit, self.omega_limit))

        old_x, old_y, old_th = self.pose

        # compute current
        current = np.zeros(2)
        if self.current_enabled:
            mag     = self.cur_strength * math.sin(2*math.pi * self.time / self.cur_period)
            current = mag * np.array([math.cos(self.cur_direction),
                                    math.sin(self.cur_direction)])

        # thrust in body frame → world frame
        F_body   = np.array([v_cmd, lat_cmd])
        R        = np.array([[math.cos(old_th), -math.sin(old_th)],
                            [math.sin(old_th),  math.cos(old_th)]])
        F_thrust = R.dot(F_body)

        # drag
        rel_v = self.vel - current
        F_drag = -self.drag_coef * rel_v if self.use_drag else np.zeros(2)

        if self.use_inertia:
            # Newtonian integration
            acc      = (F_thrust + F_drag) / self.mass
            self.vel += acc * self.dt
            dx = self.vel[0] * self.dt
            dy = self.vel[1] * self.dt
        else:
            # kinematic: no inertia, no drag, no currents
            # instantaneous velocity = thrust
            dx = F_thrust[0] * self.dt
            dy = F_thrust[1] * self.dt
            self.vel = F_thrust.copy()  # if you still want to track it

        # heading update (same in both modes)
        new_th = math.atan2(
            math.sin(old_th + omega_cmd * self.dt),
            math.cos(old_th + omega_cmd * self.dt)
        )

        new_x = old_x + dx
        new_y = old_y + dy

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
        if not self.use_continuous_reward:
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
