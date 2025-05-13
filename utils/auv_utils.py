import math
import numpy as np
import pygame


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

        # iterate each beam
        for i, rel_ang in enumerate(self.beam_angles):
            ang = heading + rel_ang
            # start at one resolution step outwards to avoid self‐collision
            r_vals = np.arange(self.resolution,
                            self.max_range + self.resolution/2,
                            self.resolution)
            for r in r_vals:
                xi = x + r * math.cos(ang)
                yi = y + r * math.sin(ang)
                gi = int(yi / self.resolution)
                gj = int(xi / self.resolution)
                # out of bounds? stop this beam
                if gi < 0 or gi >= H or gj < 0 or gj >= W:
                    break
                # hit obstacle
                if occ_grid[gi, gj]:
                    # record noisy range, mark hit
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
    

def build_maps(self):
    H,W = self.grid_size
    self.occ_grid = np.zeros((H,W),dtype=np.uint8)
    self.refl_grid = np.full((H,W),0.2)
    rectangles = [(40,40,10,60),
    (100,0,20,80),(150,120,50,10),(0,100,60,20),(80,150,10,40)]
    for cx,cy,w,h in rectangles:
        self.occ_grid[cy:cy+h,cx:cx+w] = 1
        self.refl_grid[cy:cy+h,cx:cx+w] = np.random.uniform(0.5,1.0,size=(h,w))

def build_random_maps(self):
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

def sample_random_goal(self):
    H,W=self.grid_size
    return np.array([np.random.uniform(0,W*self.resolution),
                        np.random.uniform(0,H*self.resolution)])


def sample_spawn(occ_grid: np.ndarray,
                 resolution: float,
                 start_mode: str = 'center',
                 spawn_clearance: float = 1.0) -> tuple[float, float]:
    H, W = occ_grid.shape

    if start_mode == 'center':
        return (W/2) * resolution, (H/2) * resolution

    c = int(spawn_clearance / resolution)
    frees = np.argwhere(occ_grid == 0)
    good = []
    for ry, rx in frees:
        y0min, y0max = max(0, ry - c), min(H, ry + c + 1)
        x0min, x0max = max(0, rx - c), min(W, rx + c + 1)
        if not occ_grid[y0min:y0max, x0min:x0max].any():
            good.append((ry, rx))

    if good:
        ry, rx = good[np.random.randint(len(good))]
        return (rx + 0.5) * resolution, (ry + 0.5) * resolution

    return (W/2) * resolution, (H/2) * resolution

def get_raw_observation(self):
    ranges, _, _ = self.sonar.get_readings(
        self.occ_grid, self.refl_grid, self.pose
    )
    ranges = ranges / self.sonar.max_range

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

def get_observation(self):
    # Return either the latest raw obs or the full history stack
    if self.use_history:
        return np.concatenate(self.history_buffer, axis=0)
    else:
        return self.history_buffer[-1].copy()
    

def get_cartesian_readings(self):
    ranges, _, hit_mask = self.sonar.get_readings(self.occ_grid, self.refl_grid, self.pose)
    angles = self.sonar.beam_angles + self.pose[2]
    ys = ranges * np.cos(angles)
    xs = ranges * np.sin(angles)
    local_pts = np.stack((xs, ys), axis=1)
    world_pts = local_pts + self.pose[:2]

