import numpy as np
import pygame
import sys
import math

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
    def __init__(self, grid_size=(200, 200), 
                 resolution=0.05,
                 sonar_params=None,
                current_params=None,
                docks=None,            
                dock_radius=0.2,
                dock_reward=1000,
                beacon_params=None,
                wall_thresh: float = 0.5,
                wall_penalty_coeff: float = 2.0,
                collision_penalty: float = 500.0,
                 window_size=(800, 600)):
                 
        self.grid_size = grid_size
        self.resolution = resolution
        self.window_size = window_size
        self._build_maps()
        self.wall_thresh = wall_thresh
        self.wall_penalty_coeff = wall_penalty_coeff    
        self.collision_penalty  = collision_penalty

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.previous_action = 0.0

        default_sonar = dict(
            fov=np.deg2rad(360), n_beams=60, max_range=10.0,
            resolution=resolution, noise_std=0.01,
            compute_intensity=False, spreading_loss=True,
            debris_rate=5, ghost_prob=0.05, ghost_decay=0.3
        )
        self.sonar = SonarSensor(**(sonar_params or default_sonar))

        self.current_enabled = bool(current_params)
        if self.current_enabled:
            self.cur_strength = current_params['strength']
            self.cur_period = current_params['period']
            self.cur_direction = current_params['direction']

        # Otherwise expect a list of np.array([x,y]); if None, default to 1 random dock.
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
        # ─────────────────────────────────────────────────────────────────

        # ── BEACON SETUP ────────────────────────────────────────────────
        self.beacon_enabled = bool(beacon_params)
        if self.beacon_enabled:
            self.ping_interval    = beacon_params['ping_interval']
            self.pulse_duration   = beacon_params['pulse_duration']
            self.beacon_intensity = beacon_params.get('beacon_intensity', 1.0)
            self.ping_noise       = beacon_params.get('ping_noise', 0.01)
        # ─────────────────────────────────────────────────────────────────

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
  
    def _get_obs(self):
        ranges, intensities, hit_mask = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )

        heading = np.array([np.cos(self.pose[2]), np.sin(self.pose[2])])

        # per-dock distance & bearing
        dock_feats = []
        for dock in self.docks:
            dx, dy = dock - self.pose[:2]
            dist = np.hypot(dx, dy)
            ang  = np.arctan2(dy, dx) - self.pose[2]
            ang  = np.arctan2(np.sin(ang), np.cos(ang))
            dock_feats.extend([dist, ang])

        obs = np.concatenate([
            ranges,
            intensities if intensities is not None else [],
            heading,
            dock_feats,
            [self.linear_velocity, self.angular_velocity],
            [self.previous_action]
        ])
        return obs
     

    def _get_raw_obs(self):
        ranges, intensities, hit_mask = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )
        # inject a ping from each dock if in its pulse window
        if self.beacon_enabled and (self.time % self.ping_interval) < self.pulse_duration:
            for dock in self.docks:
                dx, dy = dock - self.pose[:2]
                dist    = math.hypot(dx, dy)
                bearing = math.atan2(dy, dx) - self.pose[2]
                bearing = (bearing + math.pi) % (2*math.pi) - math.pi
                idx = np.argmin(np.abs(self.sonar.beam_angles - bearing))
                if dist < self.sonar.max_range:
                    ranges[idx]     = dist + np.random.normal(0, self.ping_noise)
                    hit_mask[idx]   = True
                    if intensities is not None:
                        intensities[idx] = self.beacon_intensity
        return ranges, intensities, hit_mask

    def reset(self):
        # 1) Place AUV at map center
        H, W = self.grid_size
        x_center = W/2 * self.resolution
        y_center = H/2 * self.resolution
        self.pose = np.array([x_center, y_center, 0.0])

        # 2) Reset motion state
        self.linear_velocity  = 0.0
        self.angular_velocity = 0.0
        self.previous_action  = 0.0

        # 3) Clear visited flags for all docks
        self._visited = [False] * len(self.docks)

        # 4) Point the AUV toward the first dock
        first = self.docks[0]
        dx, dy = first - self.pose[:2]
        self.pose[2] = np.arctan2(dy, dx)

        # ── Initialize distance‐to‐goal for shaping ────────────────────
        dists = [
            np.linalg.norm(self.pose[:2] - d)
            for idx, d in enumerate(self.docks)
            if not self._visited[idx]
        ]
        self._last_dist_to_goal = min(dists) if dists else 0.0
        # ────────────────────────────────────────────────────────────────

        # 5) Reset internal clock
        self.time = 0.0

        # 6) Return initial observation
        return self._get_obs()

    def step(self, action):
        # 1) Unpack & clip
        v, omega = action
        max_v, max_omega = 1.0, np.pi/4
        v     = np.clip(v,       -max_v,     max_v)
        omega = np.clip(omega, -max_omega, max_omega)
        self.linear_velocity, self.angular_velocity = v, omega

        # 2) Integrate orientation & position
        self.pose[2] = np.arctan2(
            np.sin(self.pose[2] + omega),
            np.cos(self.pose[2] + omega)
        )
        self.pose[0] += v * math.cos(self.pose[2])
        self.pose[1] += v * math.sin(self.pose[2])

        # 3) Collision?
        cx = int(np.clip(self.pose[0], 0, self.grid_size[1]*self.resolution-1e-5)
                 / self.resolution)
        cy = int(np.clip(self.pose[1], 0, self.grid_size[0]*self.resolution-1e-5)
                 / self.resolution)
        collision = bool(self.occ_grid[cy, cx])

        # 4) Base reward: docks
        reward = 0.0
        for i, dock in enumerate(self.docks):
            if not self._visited[i]:
                if np.linalg.norm(self.pose[:2] - dock) < self.dock_radius:
                    self._visited[i] = True
                    reward += self.dock_reward

        # 5) Shaping: distance penalty
        unvisited = [
            np.linalg.norm(self.pose[:2] - d)
            for i, d in enumerate(self.docks) if not self._visited[i]
        ]
        if unvisited:
            reward += -0.2 * min(unvisited)

        # ── Progress bonus ─────────────────────────────────────────────
        if unvisited:
            dist = min(unvisited)
            delta = self._last_dist_to_goal - dist
            if delta > 0:
                reward += 0.2 * delta
            self._last_dist_to_goal = dist
        # ────────────────────────────────────────────────────────────────

        # 6) Action cost & time penalty
        reward += -0.3 * (abs(v) + abs(omega))
        reward += -0.05 

        # 7) Wall‐proximity penalty
        ranges, _, _ = self.sonar.get_readings(self.occ_grid,
                                               self.refl_grid,
                                               self.pose)
        min_r = float(np.min(ranges))
        if min_r < self.wall_thresh:
            reward -= self.wall_penalty_coeff * (self.wall_thresh - min_r)

        # 8) Collision penalty & termination
        if collision:
            reward -= self.collision_penalty
            done = True
        else:
            done = all(self._visited)

        # 9) Record last action & advance time
        self.previous_action = v
        self.time += 1.0

        # 10) Return obs, reward, done, info
        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self):
        surf = pygame.display.set_mode(self.window_size)
        surf.fill((0, 0, 50))

        # --- 1) Draw the occupancy map ---
        map_w = 600
        total_w, total_h = self.window_size
        panel_w = (total_w - map_w) // 2
        map_h = total_h
        cw = map_w / self.grid_size[1]
        ch = map_h / self.grid_size[0]

        # draw walls
        for y, x in zip(*np.where(self.occ_grid)):
            pygame.draw.rect(surf, (100,100,100),
                            (x*cw, y*ch, cw, ch))

        # draw each dock (unvisited in yellow, visited in cyan)
        for idx, dock in enumerate(self.docks):
            gx = dock[0] / self.resolution * cw
            gy = dock[1] / self.resolution * ch
            color = (255,255,0) if not self._visited[idx] else (0,255,255)
            pygame.draw.circle(
                surf, color,
                (int(gx), int(gy)),
                int(self.dock_radius / self.resolution * cw), 2
            )

        # draw the AUV
        x_pix = self.pose[0] / self.resolution * cw
        y_pix = self.pose[1] / self.resolution * ch
        pygame.draw.circle(surf, (0,255,0), (int(x_pix), int(y_pix)), 5)
        # heading line
        ex = x_pix + 20 * math.cos(self.pose[2])
        ey = y_pix + 20 * math.sin(self.pose[2])
        pygame.draw.line(surf, (0,255,0), (int(x_pix), int(y_pix)), (int(ex), int(ey)), 2)

        # --- get beams with dock beacons baked in ---
        ranges, intensities, hit_mask = self._get_raw_obs()

        # --- 2) Fan‐beam Sonar Panel ---
        sx0 = map_w
        sw  = panel_w
        pygame.draw.rect(surf, (20,20,80), (sx0, 0, sw, total_h))
        bs = sw / self.sonar.n_beams
        for i, (r, inten, hit) in enumerate(zip(
                ranges,
                intensities if intensities is not None else [None]*len(ranges),
                hit_mask
            )):
            if not hit:
                continue
            px = sx0 + i*bs + bs/2
            py = total_h - (r/self.sonar.max_range)*(total_h-20)
            if inten is not None:
                norm = min(max(inten,0),1)
                col = (int(norm*255), 0, int((1-norm)*255))
            else:
                col = (0,200,200)
            pygame.draw.circle(surf, col, (int(px), int(py)), 6)

        # --- 3) Cartesian Sonar Display ---
        cx0 = map_w + panel_w
        cy0 = 0
        cw2 = panel_w
        ch2 = total_h
        pygame.draw.rect(surf, (30,30,30), (cx0, cy0, cw2, ch2))
        center_x = cx0 + cw2//2
        center_y = cy0 + ch2//2
        scale = cw2 / self.sonar.max_range

        for i, (r, rel_ang) in enumerate(zip(ranges, self.sonar.beam_angles)):
            dy = r * math.cos(rel_ang)
            dx = r * math.sin(rel_ang)
            px = center_x + dx * scale
            py = center_y - dy * scale
            if hit_mask[i]:
                col, rad = (255,255,0), 3
            else:
                col, rad = (50,50,50), 2
            pygame.draw.circle(surf, col, (int(px), int(py)), rad)

        pygame.display.flip()

    def get_cartesian_readings(self):
        """
        Returns sonar returns as (local_pts, world_pts, hit_mask):
        - local_pts: N×2 array of (dx, dy) in the robot frame
        - world_pts: N×2 array of absolute map-frame positions
        - hit_mask: boolean mask of valid hits
        """
        ranges, intensities, hit_mask = self._get_obs()
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
