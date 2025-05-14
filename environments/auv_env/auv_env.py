import numpy as np
import math

import pygame

from .auv_utils import (
    SonarSensor, build_maps, build_random_maps,
    sample_spawn, sample_random_goal,
    decode_action, propose_pose, check_collision,
    commit_pose, compute_base_reward, shape_reward,
    get_next_observation, get_raw_observation
)

from utils.shared_utils import (HistoryBuffer)

from typing import Optional
from .auv_config import AUVEnvConfig
from .auv_constants import DEFAULT_DISCRETE_ACTIONS, DEFAULT_SONAR_PARAMS


class AUVEnv:
    def __init__(self, cfg: Optional[AUVEnvConfig] = None, **cfg_kwargs):
        # wrap or create config
        self.cfg = cfg or AUVEnvConfig(**cfg_kwargs)

        # unpack config values
        self.grid_size       = self.cfg.grid_size
        self.window_size     = self.cfg.window_size
        self.resolution      = self.cfg.resolution

        self.random_map      = self.cfg.random_map
        self.map_fill_prob   = self.cfg.map_fill_prob
        self.smooth_steps    = self.cfg.smooth_steps
        self.birth_limit     = self.cfg.birth_limit
        self.death_limit     = self.cfg.death_limit

        self.n_beams         = self.cfg.n_beams
        self.start_mode      = self.cfg.start_mode
        self.spawn_clearance = self.cfg.spawn_clearance

        # build map once
        if self.random_map:
            build_random_maps(self)
        else:
            build_maps(self)

        # docks
        docks = self.cfg.docks
        if isinstance(docks, int):
            self.docks = [sample_random_goal(self) for _ in range(docks)]
        else:
            self.docks = docks or [sample_random_goal(self)]
        self.dock_radius = self.cfg.dock_radius
        self.dock_reward = self.cfg.dock_reward
        self._visited    = [False] * len(self.docks)

        # sonar sensor
        params = {**DEFAULT_SONAR_PARAMS, **(self.cfg.sonar_params or {})}
        params.update(n_beams=self.n_beams, resolution=self.resolution)
        self.sonar = SonarSensor(**params)

        # history buffer
        self.use_history     = self.cfg.use_history
        self.history_length  = self.cfg.history_length
        self.history_buffer  = HistoryBuffer(self.history_length + 1)

        # actions
        self.use_discrete_actions = self.cfg.use_discrete_actions
        if self.use_discrete_actions:
            self.actions = DEFAULT_DISCRETE_ACTIONS.copy()
        else:
            self.actions = None

        # reward shaping
        self.wall_thresh        = self.cfg.wall_thresh
        self.wall_penalty_coeff = self.cfg.wall_penalty_coeff
        self.collision_penalty  = self.cfg.collision_penalty
        self.progress_coeff     = self.cfg.progress_coeff
        self.turn_penalty_coeff = self.cfg.turn_penalty_coeff

        # internal state placeholders
        self._last_dist = None
        self.pose       = None

        # kickoff first episode
        self.reset()

    def reset(self) -> tuple[np.ndarray, dict]:
        x0, y0 = sample_spawn(
            occ_grid=self.occ_grid,
            resolution=self.resolution,
            start_mode=self.start_mode,
            spawn_clearance=self.spawn_clearance
        )
        self.pose = np.array([x0, y0, 0.0], dtype=float)
        first = self.docks[0]
        self.pose[2]   = math.atan2(first[1] - y0, first[0] - x0)
        self._visited  = [False] * len(self.docks)
        self._last_dist = np.linalg.norm(self.pose[:2] - first)

        ranges_and_dock = get_raw_observation(self)  
        if self.use_history:
            obs = self.history_buffer.reset(ranges_and_dock)
        else:
            obs = ranges_and_dock.copy()

        return obs, {}

    def step(self, action):
        v, omega = decode_action(action, self.actions, self.use_discrete_actions)

        old_pose, new_pose = propose_pose(self.pose, v, omega)

        collided = check_collision(old_pose, new_pose, self.occ_grid, self.resolution)

        self.pose = commit_pose(old_pose, new_pose, collided)

        reward, done = compute_base_reward(
            self.pose, self.docks, self.dock_radius,
            self.collision_penalty, self.dock_reward
        )

        reward = shape_reward(self, reward, omega)

        obs = get_next_observation(self)
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
                arr = pygame.surfarray.array3d(surf)   # (w,h,3)
                return np.transpose(arr, (1,0,2))      # (h,w,3)



