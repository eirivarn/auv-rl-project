import numpy as np
import pygame
import math
from collections import deque
from gym import spaces

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
                 dock_reward=1.0,
                 use_history: bool = False,
                 history_length: int = 3,
                 window_size=(800, 600),
                 n_beams: int = 8,
                 start_mode: str = 'center',
                 spawn_clearance: float = 1.0,
                 random_map: bool = False,
                 map_fill_prob: float = 0.3,
                 smooth_steps: int = 5,
                 birth_limit: int = 6,
                 death_limit: int = 4,
                 discrete_actions: bool = True
                ):
        # core settings
        self.grid_size = grid_size
        self.resolution = resolution
        self.window_size = window_size
        self.random_map = random_map
        # CA parameters
        self.map_fill_prob = map_fill_prob
        self.smooth_steps = smooth_steps
        self.birth_limit = birth_limit
        self.death_limit = death_limit
        
        self.start_mode = start_mode
        self.spawn_clearance = spawn_clearance

        # penalties and shaping
        self.wall_thresh = 0.5
        self.wall_penalty_coeff = 2.0
        self.collision_penalty = -1.0
        self.progress_coeff = 5.0

        # history buffer
        self.use_history = use_history
        self.history_length = history_length
        self._history_buffer = deque(maxlen=history_length+1)

        # build occupancy & reflectivity
        if self.random_map:
            self._build_random_maps()
        else:
            self._build_maps()

        # sonar
        default_sonar = dict(
            fov=np.deg2rad(360),
            n_beams=n_beams,
            max_range=10.0,
            resolution=resolution,
            noise_std=0.0,
            compute_intensity=False,
            debris_rate=0,
            ghost_prob=0.0
        )

        params = default_sonar.copy()
        if sonar_params:
            params.update(sonar_params)
        self.sonar = SonarSensor(**params)

        # actions
        self.turn_penalty_coeff = 0.5


        self.discrete_actions = discrete_actions
        if self.discrete_actions:
            self.actions = [(0.3,0.0),
                            (0.3,0.3),
                            (0.3,-0.3),
                            (0.0,0.3),
                            (0.0,-0.3)]
        else:
            # continuous forward velocity ∈ [–1,1], yaw rate ∈ [–π/4, π/4]
            self.actions = None
            low  = np.array([-1.0, -np.pi/4], dtype=np.float32)
            high = np.array([ 1.0,  np.pi/4], dtype=np.float32)
            self.action_space = spaces.Box(low, high, dtype=np.float32)

        # now that action_space is set, build an example obs to infer its shape
        example = self.reset()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=example.shape,
            dtype=np.float32
        )

        # docks
        if isinstance(docks, int):
            self.docks = [self._sample_random_goal() for _ in range(docks)]
        else:
            self.docks = docks or [self._sample_random_goal()]

        self.dock_radius = dock_radius
        self.dock_reward = dock_reward
        self._visited = [False]*len(self.docks)
        self.goal_color = (255,255,0)


        self.reset()

    def _build_maps(self):
        H,W = self.grid_size
        self.occ_grid = np.zeros((H,W),dtype=np.uint8)
        self.refl_grid = np.full((H,W),0.2)
        rectangles = [(40,40,10,60),(100,0,20,80),(150,120,50,10),(0,100,60,20),(80,150,10,40)]
        for cx,cy,w,h in rectangles:
            self.occ_grid[cy:cy+h,cx:cx+w] = 1
            self.refl_grid[cy:cy+h,cx:cx+w] = np.random.uniform(0.5,1.0,size=(h,w))

    def _build_random_maps(self):
        H,W = self.grid_size
        grid = (np.random.rand(H,W) < self.map_fill_prob).astype(np.uint8)
        def count_walls(y,x):
            total=0
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    if dy==0 and dx==0: continue
                    ny, nx = y+dy, x+dx
                    if 0<=ny<H and 0<=nx<W:
                        total+=grid[ny,nx]
                    else: total+=1
            return total
        for _ in range(self.smooth_steps):
            newg=np.zeros_like(grid)
            for y in range(H):
                for x in range(W):
                    walls=count_walls(y,x)
                    if grid[y,x]==1:
                        newg[y,x]=1 if walls>=self.death_limit else 0
                    else:
                        newg[y,x]=1 if walls>=self.birth_limit else 0
            grid=newg
        self.occ_grid=grid
        self.refl_grid=np.full((H,W),0.2,dtype=np.float32)

    def _sample_random_goal(self):
        H,W=self.grid_size
        return np.array([np.random.uniform(0,W*self.resolution),
                         np.random.uniform(0,H*self.resolution)])


    def _get_raw_obs(self):
        # 1) normalized sonar: [0…1]
        ranges, _, _ = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )
        ranges = ranges / self.sonar.max_range

        # 2) per‐dock [distance (normalized), bearing (sin,cos)]
        dock_feats = []
        for dock in self.docks:
            dx, dy = dock - self.pose[:2]
            dist = math.hypot(dx, dy) / (math.hypot(*self.grid_size)*self.resolution)
            ang  = math.atan2(dy, dx) - self.pose[2]
            dock_feats.extend([dist, math.sin(ang), math.cos(ang)])

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
        H, W = self.grid_size

        # 1) Determine spawn: center or random‐but‐clear
        if self.start_mode == 'center':
            x0 = (W/2) * self.resolution
            y0 = (H/2) * self.resolution
        else:
            # convert clearance to grid cells
            c = int(self.spawn_clearance / self.resolution)
            frees = np.argwhere(self.occ_grid == 0)
            # filter to only those free cells whose neighbourhood
            # of radius c contains no occupied cell
            good = []
            for (ry, rx) in frees:
                y0min = max(0, ry - c)
                y0max = min(H, ry + c + 1)
                x0min = max(0, rx - c)
                x0max = min(W, rx + c + 1)
                if not self.occ_grid[y0min:y0max, x0min:x0max].any():
                    good.append((ry, rx))
            if len(good):
                ry, rx = good[np.random.randint(len(good))]
                x0 = (rx + 0.5) * self.resolution
                y0 = (ry + 0.5) * self.resolution
            else:
                # fallback to center if no clear spot found
                x0 = (W/2) * self.resolution
                y0 = (H/2) * self.resolution

        # 2) Set heading, flags, history, etc. (rest unchanged)
        th0 = 0.0
        self.pose = np.array([x0, y0, th0], dtype=float)
        self._visited = [False]*len(self.docks)
        first = self.docks[0]
        self.pose[2] = math.atan2(first[1]-y0, first[0]-x0)
        self._last_dist = np.linalg.norm(self.pose[:2] - first)

        # seed history buffer
        raw0 = self._get_raw_obs()
        self._history_buffer.clear()
        for _ in range(self.history_length+1):
            self._history_buffer.append(raw0.copy())

        return self._get_obs()

    def step(self, action):
            # ─── 1) Decode & clip ───────────────────────────────────────
            if self.discrete_actions:
                v, omega = self.actions[int(action)]
            else:
                v, omega = action
            v     = float(np.clip(v,      -1.0, 1.0))
            omega = float(np.clip(omega, -np.pi/4, np.pi/4))

            # ─── 2) Propose new pose ────────────────────────────────────
            old_x, old_y, old_th = self.pose
            new_th = math.atan2(
                math.sin(old_th + omega),
                math.cos(old_th + omega)
            )
            new_x = old_x + v * math.cos(new_th)
            new_y = old_y + v * math.sin(new_th)

            # ─── 3) Continuous collision check ──────────────────────────
            dx, dy  = new_x - old_x, new_y - old_y
            dist    = math.hypot(dx, dy)
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

            # ─── 4) Commit pose ─────────────────────────────────────────
            if collided:
                # block translation, keep heading
                self.pose = np.array([old_x, old_y, new_th], dtype=float)
            else:
                self.pose = np.array([new_x, new_y, new_th], dtype=float)

            # ─── 5) Base reward & done ──────────────────────────────────
            d = np.linalg.norm(self.pose[:2] - self.docks[0])
            if collided:
                reward, done = self.collision_penalty, False
            elif d < self.dock_radius:
                reward, done = +self.dock_reward, True
            else:
                reward, done = -1.0,         False

            # ─── 6) Proximity shaping ───────────────────────────────────
            raw_ranges, _, _ = self.sonar.get_readings(
                self.occ_grid, self.refl_grid, self.pose
            )
            min_r = raw_ranges.min()
            if min_r < self.wall_thresh:
                reward -= self.wall_penalty_coeff * (1 - min_r/self.wall_thresh)

            # ─── 7) Progress shaping ────────────────────────────────────
            delta = self._last_dist - d
            reward += delta * self.progress_coeff
            self._last_dist = d

            # ─── 8) Turn penalty ────────────────────────────────────────
            reward -= self.turn_penalty_coeff * abs(omega)

            # ─── 9) Build next obs ──────────────────────────────────────
            raw = self._get_raw_obs()
            if self.use_history:
                self._history_buffer.append(raw.copy())
                obs = np.concatenate(self._history_buffer, axis=0)
            else:
                obs = raw

            return obs, reward, done, {}

    def render(self, mode='human'):
        """
        If mode=='human', draw to the screen as before.
        If mode=='rgb_array', draw into an offscreen surface and return an H×W×3 array.
        """
        # choose the target surface
        if mode == 'human':
            surf = pygame.display.set_mode(self.window_size)
        elif mode == 'rgb_array':
            # offscreen surface for headless capture
            surf = pygame.Surface(self.window_size)
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

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

        # draw the AUV
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

        if mode == 'human':
            pygame.display.flip()
            return None
        else:
            # headless: get RGB array
            arr = pygame.surfarray.array3d(surf)   # (w, h, 3)
            return np.transpose(arr, (1, 0, 2))    # (h, w, 3)


    def get_cartesian_readings(self):
        ranges, _, hit_mask = self.sonar.get_readings(self.occ_grid, self.refl_grid, self.pose)
        angles = self.sonar.beam_angles + self.pose[2]
        ys = ranges * np.cos(angles)
        xs = ranges * np.sin(angles)
        local_pts = np.stack((xs, ys), axis=1)
        world_pts = local_pts + self.pose[:2]