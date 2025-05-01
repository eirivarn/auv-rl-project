import numpy as np
import math
from environments.simple_env import simpleAUVEnv

class realisticAUVEnv(simpleAUVEnv):
    """
    simpleAUVEnv + Newtonian dynamics, drag, and currents
    """

    def __init__(self,
                 mass: float = 1.0,
                 drag_coef: float = 0.1,
                 current_params: dict = None,
                 dt: float = 0.1,
                 **kwargs):
        # --- 0) Pre-declare state so subclass-reset won't crash ---
        self.vel  = np.zeros(2, dtype=float)
        self.time = 0.0

        # --- 1) Call parent __init__ (which invokes reset()) ---
        super().__init__(**kwargs)

        # --- 2) Now override/assign physics params ---
        self.mass      = mass
        self.drag_coef = drag_coef
        self.dt        = dt

        if current_params is not None:
            self.current_enabled = True
            self.cur_strength    = current_params.get('strength', 0.2)
            self.cur_period      = current_params.get('period', 30.0)
            self.cur_direction   = current_params.get('direction', 0.0)
        else:
            self.current_enabled = False

    def reset(self):
        # 3) Let the base reset set up pose, maps, history, etc.
        obs = super().reset()
        # 4) Now zero out our dynamics state
        self.vel[:] = 0.0
        self.time   = 0.0
        return obs

    def step(self, action):
        # 1) decode commanded thrust and turn rate
        if self.discrete_actions:
            v_cmd, omega_cmd = self.actions[int(action)]
        else:
            v_cmd, omega_cmd = action
        # cap commands
        v_cmd     = float(np.clip(v_cmd,     -1.0, 1.0))
        omega_cmd = float(np.clip(omega_cmd, -np.pi/4, np.pi/4))

        old_x, old_y, old_th = self.pose

        # 2) compute ocean current at this time
        if self.current_enabled:
            mag = self.cur_strength * math.sin(2*math.pi * self.time / self.cur_period)
            current = mag * np.array([math.cos(self.cur_direction),
                                      math.sin(self.cur_direction)])
        else:
            current = np.zeros(2, dtype=float)

        # 3) thrust force in body frame -> world frame
        # assume thrust proportional to v_cmd
        F_thrust_body = np.array([v_cmd, 0.0], dtype=float)
        R = np.array([[math.cos(old_th), -math.sin(old_th)],
                      [math.sin(old_th),  math.cos(old_th)]], dtype=float)
        F_thrust = R.dot(F_thrust_body)

        # 4) drag force: proportional to relative velocity
        rel_v = self.vel - current
        F_drag = - self.drag_coef * rel_v

        # 5) integrate acceleration and velocity
        acc = (F_thrust + F_drag) / self.mass
        self.vel += acc * self.dt

        # 6) compute new candidate position
        new_x = old_x + self.vel[0] * self.dt
        new_y = old_y + self.vel[1] * self.dt

        # 7) update orientation by omega_cmd
        new_th = math.atan2(
            math.sin(old_th + omega_cmd * self.dt),
            math.cos(old_th + omega_cmd * self.dt)
        )

        # 8) collision check along line old->new
        dx, dy = new_x - old_x, new_y - old_y
        dist   = math.hypot(dx, dy)
        n_steps = max(1, int(dist / (self.resolution * 0.3)))
        collided = False
        for i in range(1, n_steps+1):
            xi = old_x + dx * (i / n_steps)
            yi = old_y + dy * (i / n_steps)
            ci = int(np.clip(xi / self.resolution, 0, self.grid_size[1]-1))
            ri = int(np.clip(yi / self.resolution, 0, self.grid_size[0]-1))
            if self.occ_grid[ri, ci]:
                collided = True
                break

        # 9) commit pose and handle collision
        if collided:
            # bounce: zero velocity
            self.vel[:] = 0.0
            # keep heading but no translation
            self.pose[2] = new_th
        else:
            self.pose = np.array([new_x, new_y, new_th], dtype=float)

        # advance time
        self.time += self.dt

        # 10) Compute reward & done exactly as in simpleAUVEnv.step:

        # distance to dock
        d = np.linalg.norm(self.pose[:2] - self.docks[0])

        # base reward & terminal
        if collided:
            reward, done = self.collision_penalty, False
        elif d < self.dock_radius:
            reward, done = +self.dock_reward, True
        else:
            reward, done = -1.0, False

        # proximity (wall‐avoidance) shaping
        raw_ranges, _, _ = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )
        min_r = raw_ranges.min()
        if min_r < self.wall_thresh:
            reward -= self.wall_penalty_coeff * (1 - min_r/self.wall_thresh)

        # progress shaping
        delta = self._last_dist - d
        reward += delta * self.progress_coeff
        self._last_dist = d

        # turn penalty
        # (action was decoded earlier into omega_cmd)
        reward -= self.turn_penalty_coeff * abs(omega_cmd)

        # build next obs
        raw = self._get_raw_obs()
        if self.use_history:
            self._history_buffer.append(raw.copy())
            obs = np.concatenate(self._history_buffer, axis=0)
        else:
            obs = raw

        return obs, reward, done, {}
