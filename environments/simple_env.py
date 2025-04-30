import numpy as np
import pygame
import sys
import math
from collections import deque
import numpy as np, math, pygame, sys


class SonarSensor:
    """
    Simulates a forward-mounted fan-beam sonar sensor with optional intensity,
    debris, and ghost echoes, ignoring beams that exit the map.
    """
    def __init__(self,
                 fov=np.deg2rad(360), n_beams=20,
                 max_range=20.0, resolution=0.05,
                 noise_std=0.00,
                 compute_intensity=False,
                 spreading_loss=True,
                 debris_rate=0,
                 ghost_prob=0.00,
                 ghost_decay=0.0):
        
        self.fov = fov
        self.n_beams = n_beams
        self.max_range = max_range
        self.resolution = resolution
        self.noise_std = noise_std
        self.compute_intensity = compute_intensity
        self.spreading_loss = spreading_loss
        self.debris_rate = debris_rate
        self.ghost_prob = ghost_prob
        self.ghost_decay = ghost_decay
        self.beam_angles = np.linspace(-fov/2, fov/2, n_beams)
        
        

    def get_readings(self, occ_grid, refl_grid, pose):
        x, y, heading = pose
        H, W = occ_grid.shape
        ranges = np.full(self.n_beams, self.max_range)
        hit_mask = np.zeros(self.n_beams, dtype=bool)
        intensities = np.zeros(self.n_beams) if self.compute_intensity else None

        for i, rel_ang in enumerate(self.beam_angles):
            ang = heading + rel_ang
            for r in np.arange(0, self.max_range, self.resolution):
                xi = x + r * math.cos(ang)
                yi = y + r * math.sin(ang)
                gi = int(yi / self.resolution)
                gj = int(xi / self.resolution)
                if gi < 0 or gi >= H or gj < 0 or gj >= W:
                    break
                if occ_grid[gi, gj]:
                    ranges[i] = r + np.random.normal(0, self.noise_std)
                    hit_mask[i] = True
                    if self.compute_intensity:
                        base = refl_grid[gi, gj]
                        loss = r**2 if self.spreading_loss and r > 0 else 1.0
                        intensities[i] = base / loss
                    break
        return ranges, intensities, hit_mask

    def get_spurious(self, ranges, intensities, hit_mask):
        spurious = []
        for _ in range(np.random.poisson(self.debris_rate)):
            i = np.random.randint(0, self.n_beams)
            r = np.random.uniform(0, self.max_range)
            inten = np.random.uniform(0, 0.1)
            spurious.append((i, r, inten))
        if intensities is not None:
            for i, r0 in enumerate(ranges):
                if hit_mask[i] and np.random.rand() < self.ghost_prob:
                    offset = np.random.uniform(0.1, 1.0)
                    r1 = min(r0 + offset, self.max_range)
                    inten = intensities[i] * self.ghost_decay
                    spurious.append((i, r1, inten))
        return spurious
    
class simpleAUVEnv:
    def __init__(self,
                 grid_size=(200, 200),
                 resolution=0.05,
                 sonar_params=None,
                 docks=None,
                 dock_radius=0.2,
                 dock_reward=1000,
                 use_history: bool = False,
                 history_length: int = 3,
                 window_size=(800, 600)):
                 
        self.grid_size = grid_size
        self.resolution = resolution
        self.window_size = window_size
        self._build_maps()

        self.wall_thresh = 0.5
        self.wall_penalty_coeff = 2.0
        self.collision_penalty = 500.0
    
        self.use_history    = use_history
        self.history_length = history_length
        # create a buffer to hold (history_length + 1) raw-observations
        self._history_buffer = deque(maxlen=history_length + 1)

        # ── SONAR SETUP ───────────────────────────────────────────────
        # we only care about ranges, no intensities or ghost echoes:
        default_sonar = dict(
            fov=np.deg2rad(360),
            n_beams=20,
            max_range=10.0,
            resolution=resolution,
            noise_std=0.0,
            compute_intensity=False,
            debris_rate=0,
            ghost_prob=0.0
        )
        self.sonar = SonarSensor(**(sonar_params or default_sonar))

        # ── DOCKS SETUP ──────────────────────────────────────────────
        if isinstance(docks, int):
            self.docks = [self._sample_random_goal() for _ in range(docks)]
        else:
            self.docks = docks or [self._sample_random_goal()]

        self.dock_radius = dock_radius
        self.dock_reward = dock_reward
        # visited flags for each dock
        self._visited   = [False] * len(self.docks)
        # color for drawing docks
        self.goal_color = (255, 255, 0)
        self.reset()

    def _build_maps(self):
        H, W = self.grid_size
        self.occ_grid = np.zeros((H, W), dtype=np.uint8)
        self.refl_grid = np.full((H, W), 0.2)
        rectangles = [
            (40, 40, 10, 60),
            (100, 0, 20, 80),
            (150, 120, 50, 10),
            (0, 100, 60, 20),
            (80, 150, 10, 40)
        ]
        for cx, cy, w, h in rectangles:
            self.occ_grid[cy:cy+h, cx:cx+w] = 1
            self.refl_grid[cy:cy+h, cx:cx+w] = np.random.uniform(0.5, 1.0, size=(h, w))
    
    def _sample_random_goal(self):
        H,W = self.grid_size
        x = np.random.uniform(0, W*self.resolution)
        y = np.random.uniform(0, H*self.resolution)
        return np.array([x,y])
  
    def _get_raw_obs(self):
        """
        Simplified observation:
          - sonar ranges: shape (n_beams,)
          - per‐dock [distance, bearing]: shape (2*len(self.docks),)
        """
        # 1) only take ranges
        ranges, _, _ = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )

        # 2) per‐dock distance & bearing
        dock_feats = []
        for dock in self.docks:
            dx, dy = dock - self.pose[:2]
            dist = math.hypot(dx, dy)
            ang  = math.atan2(dy, dx) - self.pose[2]
            ang  = (ang + math.pi) % (2*math.pi) - math.pi
            dock_feats.extend([dist, ang])

        # 3) build flat vector
        return np.concatenate([
            ranges.astype(np.float32),
            np.array(dock_feats, dtype=np.float32)
        ], axis=0)


    def _get_obs(self):
        # Return either the latest raw obs or the full history stack
        if self.use_history:
            return np.concatenate(self._history_buffer, axis=0)
        else:
            return self._history_buffer[-1].copy()
    
    def reset(self):
        # 1) Place AUV at map center
        H, W = self.grid_size
        x_center = (W/2) * self.resolution
        y_center = (H/2) * self.resolution
        self.pose = np.array([x_center, y_center, 0.0], dtype=float)

        # 2) Reset motion state
        self.linear_velocity  = 0.0
        self.angular_velocity = 0.0
        self.previous_action  = 0.0

        # 3) Clear visited flags
        self._visited = [False] * len(self.docks)

        # 4) Point AUV toward first dock
        first = self.docks[0]
        dx, dy = first - self.pose[:2]
        self.pose[2] = math.atan2(dy, dx)

        # 5) Initialize last‐distance for shaping
        dists = [
            np.linalg.norm(self.pose[:2] - dock)
            for idx, dock in enumerate(self.docks)
            if not self._visited[idx]
        ]
        self._last_dist_to_goal = min(dists) if dists else 0.0

        # 6) Reset internal clock
        self.time = 0.0

        # 7) Seed the history buffer with the initial raw observation
        raw0 = self._get_raw_obs()
        self._history_buffer.clear()
        for _ in range(self.history_length+1):
            self._history_buffer.append(raw0.copy())

        return self._get_obs()
    
    def step(self, action):
        # 1) Unpack & clip action
        v, omega = action
        max_v, max_omega = 1.0, np.pi/4
        v     = np.clip(v,       -max_v,     max_v)
        omega = np.clip(omega, -max_omega, max_omega)
        self.linear_velocity, self.angular_velocity = v, omega

        # 2) Integrate orientation & position
        self.pose[2] = math.atan2(
            math.sin(self.pose[2] + omega),
            math.cos(self.pose[2] + omega)
        )
        self.pose[0] += v * math.cos(self.pose[2])
        self.pose[1] += v * math.sin(self.pose[2])

        # 3) Collision detection
        cx = int(np.clip(self.pose[0],
                         0, self.grid_size[1]*self.resolution - 1e-5)
                 / self.resolution)
        cy = int(np.clip(self.pose[1],
                         0, self.grid_size[0]*self.resolution - 1e-5)
                 / self.resolution)
        collision = bool(self.occ_grid[cy, cx])

        # 4) Reward from visiting docks
        reward = 0.0
        for i, dock in enumerate(self.docks):
            if not self._visited[i]:
                if np.linalg.norm(self.pose[:2] - dock) < self.dock_radius:
                    self._visited[i] = True
                    reward += self.dock_reward

        # 5) Distance‐based shaping
        unvisited = [
            np.linalg.norm(self.pose[:2] - dock)
            for i, dock in enumerate(self.docks) if not self._visited[i]
        ]
        if unvisited:
            # negative cost proportional to distance
            reward += -0.2 * min(unvisited)
            # bonus for getting closer since last step
            dist = min(unvisited)
            delta = self._last_dist_to_goal - dist
            if delta > 0:
                reward += 0.2 * delta
            self._last_dist_to_goal = dist

        # 6) Action & time penalty
        reward += -0.3 * (abs(v) + abs(omega))
        reward += -0.05

        # 7) Wall‐proximity penalty via sonar
        ranges, _, _ = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )
        min_r = float(np.min(ranges))
        if min_r < self.wall_thresh:
            reward -= self.wall_penalty_coeff * (self.wall_thresh - min_r)

        # 8) Collision & termination
        if collision:
            reward -= self.collision_penalty
            done = True
        else:
            done = all(self._visited)

        # 9) Record previous action & advance time
        self.previous_action = v
        self.time += 1.0

        # 10) Build the next observation, updating history if needed
        raw = self._get_raw_obs()
        if self.use_history:
            self._history_buffer.append(raw.copy())
            obs = np.concatenate(self._history_buffer, axis=0)
        else:
            obs = raw

        return obs, reward, done, {}

    def render(self):
        surf = pygame.display.set_mode(self.window_size)
        surf.fill((0, 0, 50))

        # --- 1) Draw the occupancy map ---
        map_w = 600
        total_w, total_h = self.window_size
        panel_w = (total_w - map_w) // 2
        cw = map_w / self.grid_size[1]
        ch = total_h / self.grid_size[0]

        # draw walls
        for y, x in zip(*np.where(self.occ_grid)):
            pygame.draw.rect(surf, (100,100,100),
                             (x*cw, y*ch, cw, ch))

        # draw docks
        for idx, dock in enumerate(self.docks):
            gx = dock[0] / self.resolution * cw
            gy = dock[1] / self.resolution * ch
            color = (255,255,0) if not self._visited[idx] else (0,255,255)
            pygame.draw.circle(
                surf, color,
                (int(gx), int(gy)),
                int(self.dock_radius / self.resolution * cw), 2
            )

        # draw AUV
        x_pix = self.pose[0] / self.resolution * cw
        y_pix = self.pose[1] / self.resolution * ch
        pygame.draw.circle(surf, (0,255,0), (int(x_pix), int(y_pix)), 5)
        # heading line
        ex = x_pix + 20 * math.cos(self.pose[2])
        ey = y_pix + 20 * math.sin(self.pose[2])
        pygame.draw.line(surf, (0,255,0),
                         (int(x_pix), int(y_pix)),
                         (int(ex), int(ey)), 2)

        # --- 2) Sonar readings for the panels ---
        # get raw sonar data: we only need ranges and hit_mask here
        ranges, _, hit_mask = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )

        # 2a) Fan‐beam Sonar Panel
        sx0 = map_w
        sw  = panel_w
        pygame.draw.rect(surf, (20,20,80), (sx0, 0, sw, total_h))
        bs = sw / len(ranges)
        for i, (r, hit) in enumerate(zip(ranges, hit_mask)):
            if not hit: continue
            px = sx0 + i*bs + bs/2
            py = total_h - (r/self.sonar.max_range)*(total_h-20)
            pygame.draw.circle(surf, (0,200,200), (int(px), int(py)), 6)

        # 2b) Cartesian Sonar Display
        cx0 = map_w + panel_w
        cw2 = panel_w
        ch2 = total_h
        pygame.draw.rect(surf, (30,30,30), (cx0, 0, cw2, ch2))
        center_x = cx0 + cw2//2
        center_y = ch2//2
        scale = cw2 / self.sonar.max_range

        for i, (r, rel_ang) in enumerate(zip(ranges, self.sonar.beam_angles)):
            dy = r * math.cos(rel_ang)
            dx = r * math.sin(rel_ang)
            px = center_x + dx * scale
            py = center_y - dy * scale
            col, rad = ((255,255,0), 3) if hit_mask[i] else ((50,50,50), 2)
            pygame.draw.circle(surf, col, (int(px), int(py)), rad)

        pygame.display.flip()



    def get_cartesian_readings(self):
        """
        Returns sonar returns as (local_pts, world_pts, hit_mask):
        - local_pts: N×2 array of (dx, dy) in the robot frame
        - world_pts: N×2 array of absolute map-frame positions
        - hit_mask: boolean mask of valid hits
        """
        ranges, _, hit_mask = self.sonar.get_readings(self.occ_grid, self.refl_grid, self.pose)
        angles = self.sonar.beam_angles + self.pose[2]
        ys = ranges * np.cos(angles)
        xs = ranges * np.sin(angles)
        local_pts = np.stack((xs, ys), axis=1)
        world_pts = local_pts + self.pose[:2]
        return local_pts, world_pts, hit_mask

if __name__=='__main__':
    pygame.init()
    env = simpleAUVEnv(
        sonar_params    = {'compute_intensity': True},
        current_params  = {'strength': 0.2, 'period': 30.0, 'direction': np.deg2rad(45)},
        goal_params     = {'radius': 0.5},
        beacon_params   = {'ping_interval': 1.0, 'pulse_duration': 0.1,
                            'beacon_intensity': 1.0, 'ping_noise': 0.01}
    )
    clock = pygame.time.Clock()
    obs = env.reset()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
        keys = pygame.key.get_pressed()
        v     =  1.0 if keys[pygame.K_UP]   else -1.0 if keys[pygame.K_DOWN]  else 0.0
        omega =  1.0 if keys[pygame.K_LEFT] else -1.0 if keys[pygame.K_RIGHT] else 0.0
        obs, rew, done, _ = env.step((v, omega))
        if done: obs = env.reset()
        env.render()
        clock.tick(60)
