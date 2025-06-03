import math
from typing import Tuple
import numpy as np
import pygame


class SonarSensor:
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
                  (100,0,20,80),
                  (150,120,50,10),
                  (0,100,60,20),
                  (80,150,10,40)]
    for cx,cy,w,h in rectangles:
        self.occ_grid[cy:cy+h,cx:cx+w] = 1
        self.refl_grid[cy:cy+h,cx:cx+w] = np.random.uniform(0.5,1.0,size=(h,w))

    # ── add outer wall ─────────────────────────────────────────
    self.occ_grid[ 0, :] = 1
    self.occ_grid[-1, :] = 1
    self.occ_grid[:,  0] = 1
    self.occ_grid[:, -1] = 1

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
    return random_clear_spawn(
        self.occ_grid,
        self.grid_size,
        self.resolution,
        getattr(self, 'spawn_clearance', 1.0)
    )

def center_spawn(grid_size: Tuple[int, int], resolution: float) -> Tuple[float, float]:
    H, W = grid_size
    x = (W / 2) * resolution
    y = (H / 2) * resolution
    return x, y

def random_clear_spawn(
    occ_grid: np.ndarray,
    grid_size: Tuple[int, int],
    resolution: float,
    spawn_clearance: float
) -> Tuple[float, float]:
    H, W = grid_size
    c = int(spawn_clearance / resolution)
    frees = np.argwhere(occ_grid == 0)
    valid = []
    for ry, rx in frees:
        y0min = max(0, ry - c)
        y0max = min(H, ry + c + 1)
        x0min = max(0, rx - c)
        x0max = min(W, rx + c + 1)
        if not occ_grid[y0min:y0max, x0min:x0max].any():
            valid.append((ry, rx))
    if valid:
        ry, rx = valid[np.random.randint(len(valid))]
        x = (rx + 0.5) * resolution
        y = (ry + 0.5) * resolution
        return x, y
    # fallback to center
    return center_spawn(grid_size, resolution)


def sample_spawn(
    start_mode: str,
    occ_grid: np.ndarray,
    grid_size: Tuple[int, int],
    resolution: float,
    spawn_clearance: float = 1.0
) -> Tuple[float, float]:
    if start_mode == 'center':
        return center_spawn(grid_size, resolution)
    else:
        return random_clear_spawn(occ_grid, grid_size, resolution, spawn_clearance)


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

def decode_action(action, actions, use_discrete_actions):
    """
    Decode and clip raw action into (v, omega).
    """
    if use_discrete_actions:
        v, omega = actions[int(action)]
    else:
        v, omega = action
    v     = float(np.clip(v, -1.0, 1.0))
    omega = float(np.clip(omega, -np.pi/4, np.pi/4))
    return v, omega


def propose_pose(pose, v, omega):
    """
    Compute new pose from old pose and commanded velocities.
    Returns (old_pose, new_pose).
    """
    old_x, old_y, old_th = pose
    new_th = math.atan2(math.sin(old_th + omega), math.cos(old_th + omega))
    new_x = old_x + v * math.cos(new_th)
    new_y = old_y + v * math.sin(new_th)
    return (old_x, old_y, old_th), (new_x, new_y, new_th)


def check_collision(old_pose, new_pose, occ_grid, resolution):
    """
    Continuous collision check from old to new pose.
    Returns True if collision or out‐of‐bounds occurs.
    """
    ox, oy, _ = old_pose
    nx, ny, _ = new_pose
    dx, dy = nx - ox, ny - oy
    dist = math.hypot(dx, dy)
    steps = max(1, int(dist / (resolution * 0.3)))

    H, W = occ_grid.shape
    for i in range(1, steps + 1):
        xi = ox + dx * (i/steps)
        yi = oy + dy * (i/steps)

        # convert to grid‐indices (float)
        x_idx = xi / resolution
        y_idx = yi / resolution

        # out‐of‐bounds ⇒ treat as collision
        if x_idx < 0 or x_idx >= W or y_idx < 0 or y_idx >= H:
            return True

        ri = int(y_idx)
        ci = int(x_idx)
        if occ_grid[ri, ci]:
            return True

    return False


def commit_pose(old_pose, new_pose, collided):
    """
    Return updated pose array given collision flag.
    """
    if collided:
        ox, oy, _ = old_pose
        _, _, nth = new_pose
        return np.array([ox, oy, nth], dtype=float)
    return np.array(new_pose, dtype=float)


def compute_base_reward(pose, docks, dock_radius, collision_penalty, dock_reward):
    """
    Compute base reward and done flag.
    """
    d = np.linalg.norm(pose[:2] - docks[0])
    if collision_penalty < 0:
        # assuming collision_penalty negative sign included
        return collision_penalty, False
    if d < dock_radius:
        return dock_reward, True
    return -1.0, False


def shape_reward(env, reward, omega):
    """
    Apply proximity, progress, and turn shaping.
    """
    ranges, _, _ = env.sonar.get_readings(env.occ_grid, env.refl_grid, env.pose)
    min_r = ranges.min()
    if min_r < env.wall_thresh:
        reward -= env.wall_penalty_coeff * (1 - min_r/env.wall_thresh)
    d = np.linalg.norm(env.pose[:2] - env.docks[0])
    delta = env._last_dist - d
    reward += delta * env.progress_coeff
    env._last_dist = d
    reward -= env.turn_penalty_coeff * abs(omega)
    return reward


def get_next_observation(env):
    """
    Retrieve next observation, optionally stacking history.
    """
    raw = get_raw_observation(env)
    if env.use_history:
        return env.history_buffer.process(raw)
    return raw.copy()