import numpy as np
import math

import pygame
from utils.auv_utils import (
    SonarSensor,
    build_maps,
    build_random_maps,
    sample_spawn,
    sample_random_goal,
    get_raw_observation

)
from utils.grid_utils import HistoryBuffer
from utils.auv_constants import (
    DEFAULT_AUV_GRID_SIZE,
    DEFAULT_RESOLUTION,
    DEFAULT_SONAR_PARAMS,
    DEFAULT_DOCKS,
    DEFAULT_DOCK_RADIUS,
    DEFAULT_DOCK_REWARD,
    DEFAULT_USE_HISTORY,
    DEFAULT_HISTORY_LENGTH,
    DEFAULT_N_BEAMS,
    DEFAULT_START_MODE,
    DEFAULT_SPAWN_CLEARANCE,
    DEFAULT_RANDOM_MAP,
    DEFAULT_MAP_FILL_PROB,
    DEFAULT_SMOOTH_STEPS,
    DEFAULT_BIRTH_LIMIT,
    DEFAULT_DEATH_LIMIT,
    DEFAULT_WALL_THRESH,
    DEFAULT_WALL_PENALTY_COEFF,
    DEFAULT_COLLISION_PENALTY,
    DEFAULT_PROGRESS_COEFF,
    DEFAULT_TURN_PENALTY_COEFF,
    DEFAULT_USE_DISCRETE_ACTIONS,
    DEFAULT_DISCRETE_ACTIONS,
    DEFAULT_WINDOW_SIZE
)

class AUVEnv:
    def __init__(self,
                 grid_size=DEFAULT_AUV_GRID_SIZE,
                 window_size= DEFAULT_WINDOW_SIZE,
                 resolution=DEFAULT_RESOLUTION,
                 sonar_params=DEFAULT_SONAR_PARAMS,
                 docks=DEFAULT_DOCKS,
                 dock_radius=DEFAULT_DOCK_RADIUS,
                 dock_reward=DEFAULT_DOCK_REWARD,
                 use_history: bool = DEFAULT_USE_HISTORY,
                 history_length: int = DEFAULT_HISTORY_LENGTH,
                 n_beams: int = DEFAULT_N_BEAMS,
                 start_mode: str = DEFAULT_START_MODE,
                 spawn_clearance: float = DEFAULT_SPAWN_CLEARANCE,
                 random_map: bool = DEFAULT_RANDOM_MAP,
                 map_fill_prob: float = DEFAULT_MAP_FILL_PROB,
                 smooth_steps: int = DEFAULT_SMOOTH_STEPS,
                 birth_limit: int = DEFAULT_BIRTH_LIMIT,
                 death_limit: int = DEFAULT_DEATH_LIMIT,
                 wall_thresh: float = DEFAULT_WALL_THRESH,
                 wall_penalty_coeff: float = DEFAULT_WALL_PENALTY_COEFF,
                 collision_penalty: float = DEFAULT_COLLISION_PENALTY,
                 progress_coeff: float = DEFAULT_PROGRESS_COEFF,
                 turn_penalty_coeff: float = DEFAULT_TURN_PENALTY_COEFF,
                 use_discrete_actions: bool = DEFAULT_USE_DISCRETE_ACTIONS
                ):
        # core params
        self.grid_size        = grid_size
        self.window_size      = window_size
        self.resolution       = resolution
        self.random_map       = random_map
        self.n_beams          = n_beams
        self.start_mode       = start_mode
        self.spawn_clearance  = spawn_clearance

        # map generation
        self.map_fill_prob = map_fill_prob
        self.smooth_steps   = smooth_steps
        self.birth_limit    = birth_limit
        self.death_limit    = death_limit
        if self.random_map:
            build_random_maps(self)
        else:
            build_maps(self)

        # reward shaping
        self.wall_thresh         = wall_thresh
        self.wall_penalty_coeff  = wall_penalty_coeff
        self.collision_penalty   = collision_penalty
        self.progress_coeff      = progress_coeff
        self.turn_penalty_coeff  = turn_penalty_coeff

        # docks
        if isinstance(docks, int):
            self.docks = [sample_random_goal(self) for _ in range(docks)]
        else:
            self.docks = docks or [sample_random_goal(self)]
        self.dock_radius = dock_radius
        self.dock_reward = dock_reward
        self._visited    = [False]*len(self.docks)

        # sensors
        # ─── Sonar ──────────────────────────────────
        params = DEFAULT_SONAR_PARAMS.copy()
        params.update(sonar_params or {})
        params["n_beams"]    = self.n_beams
        params["resolution"] = self.resolution
        self.sonar = SonarSensor(**params)

        # ─── History ─────────────────────────────────
        self.use_history     = use_history
        self.history_length  = history_length
        self.history_buffer  = HistoryBuffer(history_length + 1)

        # discrete actions
        self.use_discrete_actions = use_discrete_actions
        if self.use_discrete_actions:
            self.actions = DEFAULT_DISCRETE_ACTIONS.copy()
        else:
            self.actions = None

        # internal state
        self._last_dist = None
        self.pose        = None

        # seed first episode
        self.reset()

    def reset(self) -> tuple[np.ndarray, dict]:
        # 1) spawn
        x0, y0 = sample_spawn(
            occ_grid=self.occ_grid,
            resolution=self.resolution,
            start_mode=self.start_mode,
            spawn_clearance=self.spawn_clearance
        )
        # 2) pose & heading
        self.pose = np.array([x0, y0, 0.0], dtype=float)
        first = self.docks[0]
        self.pose[2]   = math.atan2(first[1] - y0, first[0] - x0)
        self._visited  = [False] * len(self.docks)
        self._last_dist = np.linalg.norm(self.pose[:2] - first)

        ranges_and_dock = get_raw_observation(self)  # = [ranges; dock_feats]
        if self.use_history:
            obs = self.history_buffer.reset(ranges_and_dock)
        else:
            obs = ranges_and_dock.copy()

        return obs, {}

    def step(self, action):
        # 1) pick v,ω
        if self.use_discrete_actions:
            v, omega = self.actions[int(action)]
        else:
            v, omega = action
        v     = float(np.clip(v,      -1.0, 1.0))
        omega = float(np.clip(omega, -np.pi/4, np.pi/4))

        # 2) propose
        old_x, old_y, old_th = self.pose
        new_th = math.atan2(math.sin(old_th+omega),
                             math.cos(old_th+omega))
        new_x = old_x + v * math.cos(new_th)
        new_y = old_y + v * math.sin(new_th)

        # 3) collision
        dx, dy = new_x-old_x, new_y-old_y
        n_steps = max(1, int(math.hypot(dx,dy)/(self.resolution*0.3)))
        collided = False
        for i in range(1, n_steps+1):
            xi = old_x + dx*(i/n_steps)
            yi = old_y + dy*(i/n_steps)
            ri = int(np.clip(yi/self.resolution, 0, self.grid_size[0]-1))
            ci = int(np.clip(xi/self.resolution, 0, self.grid_size[1]-1))
            if self.occ_grid[ri,ci]:
                collided = True
                break

        # 4) commit
        if collided:
            self.pose = np.array([old_x, old_y, new_th], dtype=float)
        else:
            self.pose = np.array([new_x, new_y, new_th], dtype=float)

        # 5) reward/done
        d = np.linalg.norm(self.pose[:2] - self.docks[0])
        if collided:
            reward, done = -self.collision_penalty, False
        elif d < self.dock_radius:
            reward, done = +self.dock_reward, True
        else:
            reward, done = -1.0,         False

        # 6) proximity shaping
        ranges, _, _ = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )
        min_r = ranges.min()
        if min_r < self.wall_thresh:
            reward -= self.wall_penalty_coeff * (1 - min_r/self.wall_thresh)

        # 7) progress shaping
        delta = self._last_dist - d
        reward += delta * self.progress_coeff
        self._last_dist = d

        # 8) turn penalty
        reward -= self.turn_penalty_coeff * abs(omega)

        # 9) next obs
        raw = get_raw_observation(self)  
        if self.use_history:
            obs = self.history_buffer.process(raw)
        else:
            obs = raw.copy()

        return obs, reward, done, {}
    
    
    def render(self, mode='human'):
            """
            mode='human': pop up a window.
            mode='rgb_array': headless offscreen.
            """
            # 1) surface setup
            if mode == 'human':
                surf = pygame.display.set_mode(self.window_size)
            elif mode == 'rgb_array':
                surf = pygame.Surface(self.window_size)
            else:
                raise ValueError("Unsupported render mode")

            total_w, total_h = self.window_size
            map_w = 600
            panel_w = (total_w - map_w) // 2

            # fill background
            surf.fill((0, 0, 50))

            # 2) draw occupancy map
            cw = map_w / self.grid_size[1]
            ch = total_h / self.grid_size[0]
            for y, x in zip(*np.where(self.occ_grid)):
                pygame.draw.rect(surf, (100,100,100),
                                (x*cw, y*ch, cw, ch))

            # 3) draw docks
            for idx, dock in enumerate(self.docks):
                gx = dock[0] / self.resolution * cw
                gy = dock[1] / self.resolution * ch
                color = (255,255,0) if not self._visited[idx] else (0,255,255)
                pygame.draw.circle(surf, color,
                                (int(gx), int(gy)),
                                int(self.dock_radius/self.resolution * cw), 2)

            # 4) draw AUV
            x_pix = self.pose[0] / self.resolution * cw
            y_pix = self.pose[1] / self.resolution * ch
            pygame.draw.circle(surf, (0,255,0),
                            (int(x_pix), int(y_pix)),
                            max(3, int(cw*0.5)))
            # heading line
            ex = x_pix + 20*math.cos(self.pose[2])
            ey = y_pix + 20*math.sin(self.pose[2])
            pygame.draw.line(surf, (0,255,0),
                            (int(x_pix),int(y_pix)),
                            (int(ex),int(ey)), 2)

            # 5) get sonar readings
            ranges, _, hit_mask = self.sonar.get_readings(
                self.occ_grid, self.refl_grid, self.pose
            )

            # 6a) fan‐beam panel
            sx0 = map_w
            sw  = panel_w
            pygame.draw.rect(surf, (20,20,80), (sx0, 0, sw, total_h))
            bs = sw / len(ranges)
            for i, (r, hit) in enumerate(zip(ranges, hit_mask)):
                px = sx0 + (i+0.5)*bs
                # scale r→vertical position (leave 10px margin)
                py = total_h - (r/self.sonar.max_range)*(total_h-20) - 10
                color = (0,200,200) if hit else (50,50,50)
                radius = 5 if hit else 2
                pygame.draw.circle(surf, color, (int(px), int(py)), radius)

            # 6b) Cartesian panel
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

            # flip or return
            if mode == 'human':
                pygame.display.flip()
                return None
            else:
                arr = pygame.surfarray.array3d(surf)   # (w,h,3)
                return np.transpose(arr, (1,0,2))      # (h,w,3)



