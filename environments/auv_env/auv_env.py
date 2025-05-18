from collections import deque
import numpy as np
import math
from gym import spaces
import pygame

from .auv_utils import (
    SonarSensor, build_maps, build_random_maps,
    sample_spawn, sample_random_goal,
    decode_action, propose_pose, check_collision,
    commit_pose, compute_base_reward, shape_reward,
    get_next_observation, get_raw_observation
)

from utils.shared_utils import HistoryBuffer

from typing import Optional
from .auv_config import AUVEnvConfig
from .auv_constants import DEFAULT_DISCRETE_ACTIONS, DEFAULT_SONAR_PARAMS


class AUVEnv:
    def __init__(self, cfg: Optional[AUVEnvConfig] = None, **cfg_kwargs):
        # wrap or create config
        self.cfg = cfg or AUVEnvConfig(**cfg_kwargs)

        # physics toggle
        self.use_physics = self.cfg.use_physics

        # physics parameters
        self.mass      = self.cfg.mass
        self.drag_coef = self.cfg.drag_coef
        self.dt        = self.cfg.dt

        # ocean currents
        cur_params = getattr(self.cfg, 'current_params', None)
        if cur_params is not None:
            self.current_enabled = True
            # allow override keys in config
            self.cur_strength  = cur_params.get('strength', cur_params.get('current_speed', 0.0))
            self.cur_period    = cur_params.get('period', 1.0)
            self.cur_direction = cur_params.get('direction', cur_params.get('current_direction', 0.0))
            self.cur_noise_std = cur_params.get('noise_std', cur_params.get('current_noise_std', 0.0))
        else:
            self.current_enabled = False

        # how often to rebuild a random map (in resets)
        self.map_reset_freq = getattr(self.cfg, 'map_reset_freq', 0)
        self._reset_count   = 0

        # unpack config values
        self.grid_size       = self.cfg.grid_size
        self.window_size     = self.cfg.window_size
        self.resolution      = self.cfg.resolution
        self.start_mode      = self.cfg.start_mode
        self.spawn_clearance = self.cfg.spawn_clearance

        self.random_map      = self.cfg.random_map
        self.map_fill_prob   = self.cfg.map_fill_prob
        self.smooth_steps    = self.cfg.smooth_steps
        self.birth_limit     = self.cfg.birth_limit
        self.death_limit     = self.cfg.death_limit

        # build map once
        if self.random_map:
            build_random_maps(self)
        else:
            build_maps(self)

        # sonar sensor
        params = {**DEFAULT_SONAR_PARAMS, **(self.cfg.sonar_params or {})}
        params.update(n_beams=self.cfg.n_beams, resolution=self.resolution)
        self.sonar = SonarSensor(**params)

        # history buffer
        self.use_history    = self.cfg.use_history
        self.history_length = self.cfg.history_length
        self.history_buffer = HistoryBuffer(self.history_length + 1)

        self.use_discrete_actions = self.cfg.use_discrete_actions
        if self.use_discrete_actions:
            # only in discrete mode do we need the lookup table
            self.actions = DEFAULT_DISCRETE_ACTIONS.copy()
        else:
            # no more self.actions in continuous mode
            self.actions = None
        # expose a gym‐style action_space for both modes
        if self.use_discrete_actions:
            self.action_space = spaces.Discrete(len(DEFAULT_DISCRETE_ACTIONS))
        else:
            # these limits should match your vehicle’s real bounds
            low  = np.array([-self.cfg.max_thrust, -self.cfg.max_torque], dtype=np.float32)
            high = np.array([ self.cfg.max_thrust,  self.cfg.max_torque], dtype=np.float32)
            self.action_space = spaces.Box(low, high, dtype=np.float32)
            
        self.dock_radius = self.cfg.dock_radius
        self.wall_thresh = self.cfg.wall_thresh

        # internal state placeholders
        self._last_dist     = None
        self.pose           = None
        self.velocity       = np.zeros(2, dtype=float)
        self.action_history = deque(maxlen=self.history_length)

        # kickoff first episode
        self.reset()

    def reset(self) -> tuple[np.ndarray, dict]:
        # reset physics state if used
        if self.use_physics:
            self.velocity[:] = 0.0

        # 1) re-sample docks in free space
        docks_cfg = self.cfg.docks
        if isinstance(docks_cfg, int):
            self.docks = [sample_random_goal(self) for _ in range(docks_cfg)]
        else:
            self.docks = docks_cfg or [sample_random_goal(self)]
        self._visited = [False] * len(self.docks)

        # 2) spawn the AUV
        x0, y0 = sample_spawn(
            occ_grid=self.occ_grid,
            grid_size=self.grid_size,
            resolution=self.resolution,
            start_mode=self.start_mode,
            spawn_clearance=self.spawn_clearance
        )
        self.pose = np.array([x0, y0, 0.0], dtype=float)
        first = self.docks[0]
        self.pose[2] = math.atan2(first[1] - y0, first[0] - x0)
        self._last_dist = np.linalg.norm(self.pose[:2] - first)

        # 3) reset action history (zeros = “no-op”)
        self.action_history = deque(maxlen=self.history_length)
        if self.use_discrete_actions:
            for _ in range(self.history_length):
                self.action_history.append(0.0)
        else:
            for _ in range(self.history_length):
                self.action_history.extend([0.0, 0.0])

        # 4) initial observation build
        ranges, _, hit_mask = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )
        ranges = ranges / self.sonar.max_range
        hits = hit_mask.astype(np.float32)
        dock_feats = []
        for dock in self.docks:
            dx, dy = dock - self.pose[:2]
            dist = math.hypot(dx, dy) / (math.hypot(*self.grid_size) * self.resolution)
            ang = math.atan2(dy, dx) - self.pose[2]
            dock_feats.extend([dist, math.sin(ang), math.cos(ang)])
        raw0 = np.concatenate([
            ranges.astype(np.float32),
            hits,
            np.array(dock_feats, dtype=np.float32),
            np.array(self.action_history, dtype=np.float32)
        ], axis=0)

        if self.use_history:
            obs = self.history_buffer.reset(raw0)
        else:
            obs = raw0.copy()

        return obs, {}

    def step(self, action):
        # decode control input
        if self.use_discrete_actions:
            # integer index → (v, ω)
            thrust, torque = decode_action(action, self.actions, True)
        else:
            # continuous: expect a 2-vector [v, ω]
            try:
                thrust, torque = action
            except Exception:
                raise ValueError(f"Expected continuous action (thrust, torque), got {action}")


        # choose motion model
        if self.use_physics:
            # physics-based motion
            force_body = np.array([thrust, 0.0], dtype=float)
            theta = self.pose[2]
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            force_world = np.array([
                cos_t * force_body[0] - sin_t * force_body[1],
                sin_t * force_body[0] + cos_t * force_body[1]
            ])
            drag_force = -self.drag_coef * self.velocity

            vel_current = np.zeros(2)
            if self.current_enabled:
                phase = 2 * math.pi * ((self._reset_count * self.dt) / self.cur_period)
                strength = self.cur_strength * math.sin(phase)
                vel_current = strength * np.array([
                    math.cos(self.cur_direction),
                    math.sin(self.cur_direction)
                ])
                # optional noise
                vel_current += np.random.randn(2) * self.cur_noise_std

            accel = (force_world + drag_force) / self.mass
            self.velocity += accel * self.dt

            new_pos = self.pose[:2] + (self.velocity + vel_current) * self.dt
            new_theta = self.pose[2] + (torque / self.mass) * self.dt

            collided = check_collision(
                np.array([*self.pose[:2], self.pose[2]]),
                np.array([*new_pos, new_theta]),
                self.occ_grid, self.resolution
            )
            if collided:
                self.velocity[:] = 0.0
            else:
                self.pose[:2] = new_pos
                self.pose[2]  = new_theta
        else:
            # legacy kinematic motion
            old_pose, new_pose = propose_pose(self.pose, thrust, torque)
            collided = check_collision(old_pose, new_pose, self.occ_grid, self.resolution)
            self.pose = commit_pose(old_pose, new_pose, collided)

        # reward & done (mode-specific)
        # --- sparse dock & collision penalty ---

        reward = 0.0
        done = False

        
        if self.use_discrete_actions:
            dock_reward   = self.cfg.discrete_dock_reward
        else:
            dock_reward   = self.cfg.continuous_dock_reward

        d = np.linalg.norm(self.pose[:2] - self.docks[0])
        if d < self.dock_radius:
             reward += dock_reward
             done = True

        # --- step cost ---
        step_c = (self.cfg.discrete_step_cost
                if self.use_discrete_actions
                else self.cfg.continuous_step_cost)
        reward += step_c

        # --- wall proximity penalty ---
        ranges, _, _ = self.sonar.get_readings(self.occ_grid,
                                            self.refl_grid,
                                            self.pose)
        min_r = ranges.min() / self.sonar.max_range
        wall_c = (self.cfg.discrete_wall_penalty_coeff
                if self.use_discrete_actions
                else self.cfg.continuous_wall_penalty_coeff)
        if min_r < self.wall_thresh:
            reward -= wall_c * (1 - min_r / self.wall_thresh)

        # --- progress reward ---
        d_new = np.linalg.norm(self.pose[:2] - self.docks[0])
        delta = self._last_dist - d_new
        prog_c = (self.cfg.discrete_progress_coeff
                if self.use_discrete_actions
                else self.cfg.continuous_progress_coeff)
        reward += prog_c * delta
        self._last_dist = d_new

        # --- turn penalty ---
        turn_c = (self.cfg.discrete_turn_penalty_coeff
                if self.use_discrete_actions
                else self.cfg.continuous_turn_penalty_coeff)
        reward -= turn_c * abs(torque)

        # build observation
        ranges, _, hit_mask = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )
        ranges = ranges / self.sonar.max_range
        hits   = hit_mask.astype(np.float32)
        dock_feats = []
        for dock in self.docks:
            dx, dy = dock - self.pose[:2]
            dist = math.hypot(dx, dy) / (math.hypot(*self.grid_size) * self.resolution)
            ang = math.atan2(dy, dx) - self.pose[2]
            dock_feats.extend([dist, math.sin(ang), math.cos(ang)])
        raw = np.concatenate([
            ranges.astype(np.float32),
            hits,
            np.array(dock_feats, dtype=np.float32),
            np.array(self.action_history, dtype=np.float32)
        ], axis=0)

        if self.use_history:
            obs = self.history_buffer.process(raw)
        else:
            obs = raw

        return obs, reward, done, {}

    def render(self, mode='human') -> np.ndarray:
        if mode == 'human':
            surf = pygame.display.set_mode(self.window_size)
        elif mode == 'rgb_array':
            surf = pygame.Surface(self.window_size)
        else:
            raise ValueError("Unsupported render mode")

        total_w, total_h = self.window_size
        map_w = 600
        panel_w = (total_w - map_w) // 2

        surf.fill((0, 0, 50))

        cw = map_w / self.grid_size[1]
        ch = total_h / self.grid_size[0]
        for y, x in zip(*np.where(self.occ_grid)):
            pygame.draw.rect(surf, (100,100,100),
                             (x*cw, y*ch, cw, ch))

        for idx, dock in enumerate(self.docks):
            gx = dock[0] / self.resolution * cw
            gy = dock[1] / self.resolution * ch
            color = (255,255,0) if not self._visited[idx] else (0,255,255)
            pygame.draw.circle(surf, color,
                               (int(gx), int(gy)),
                               int(self.dock_radius/self.resolution * cw), 2)

        x_pix = self.pose[0] / self.resolution * cw
        y_pix = self.pose[1] / self.resolution * ch
        pygame.draw.circle(surf, (0,255,0),
                           (int(x_pix), int(y_pix)),
                           max(3, int(cw*0.5)))

        ex = x_pix + 20*math.cos(self.pose[2])
        ey = y_pix + 20*math.sin(self.pose[2])
        pygame.draw.line(surf, (0,255,0),
                         (int(x_pix),int(y_pix)),
                         (int(ex),int(ey)), 2)

        ranges, _, hit_mask = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )

        sx0 = map_w
        sw  = panel_w
        pygame.draw.rect(surf, (20,20,80), (sx0, 0, sw, total_h))
        bs = sw / len(ranges)
        for i, (r, hit) in enumerate(zip(ranges, hit_mask)):
            px = sx0 + (i+0.5)*bs
            py = total_h - (r/self.sonar.max_range)*(total_h-20) - 10
            color = (0,200,200) if hit else (50,50,50)
            radius = 5 if hit else 2
            pygame.draw.circle(surf, color, (int(px), int(py)), radius)

        cx0 = map_w + panel_w
        cw2 = panel_w
        ch2 = total_h
        pygame.draw.rect(surf, (30,30,30), (cx0, 0, cw2, ch2))
        center_x = cx0 + cw2//2
        center_y = ch2//2
        scale = cw2 / self.sonar.max_range
        for i, (r, rel_ang) in enumerate(zip(ranges, self.sonar.beam_angles)):
            ang = rel_ang + self.pose[2]
            dx = r * math.sin(ang)
            dy = r * math.cos(ang)
            px = center_x + dx*scale
            py = center_y - dy*scale
            color = (255,255,0) if hit_mask[i] else (80,80,80)
            radius = 4 if hit_mask[i] else 2
            pygame.draw.circle(surf, color, (int(px), int(py)), radius)

        if mode == 'human':
            pygame.display.flip()
            return None
        else:
            arr = pygame.surfarray.array3d(surf)
            return np.transpose(arr, (1,0,2))