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
                 fov=np.deg2rad(90),
                 n_beams=60,
                 max_range=10.0,
                 resolution=0.05,
                 noise_std=0.01,
                 compute_intensity=False,
                 spreading_loss=True,
                 debris_rate=5,
                 ghost_prob=0.05,
                 ghost_decay=0.3):
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

        # Cast each beam
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
                    ranges[i] = r
                    hit_mask[i] = True
                    if self.compute_intensity:
                        base = refl_grid[gi, gj]
                        loss = r**2 if self.spreading_loss and r > 0 else 1.0
                        intensities[i] = base / loss
                    break

        if self.compute_intensity and intensities is not None:
            speck = np.random.normal(1.0, self.noise_std, size=self.n_beams)
            intensities[hit_mask] *= speck[hit_mask]
            intensities = np.clip(intensities, 0, None)

        if hit_mask.any():
            noise = np.random.normal(0, self.noise_std, size=self.n_beams)
            ranges[hit_mask] += noise[hit_mask]
            ranges = np.clip(ranges, 0, self.max_range)

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

class AUVEnv:
    def __init__(self,
                 grid_size=(200, 200),
                 resolution=0.05,
                 sonar_params=None,
                 current_params=None,
                 goal_params=None,
                 beacon_params=None,
                 window_size=(800, 600)):
        self.grid_size = grid_size
        self.resolution = resolution
        self.window_size = window_size
        self._build_maps()

        default_sonar = dict(
            fov=np.deg2rad(90), n_beams=60, max_range=10.0,
            resolution=resolution, noise_std=0.01,
            compute_intensity=False, spreading_loss=True,
            debris_rate=5, ghost_prob=0.05, ghost_decay=0.3
        )
        self.sonar = SonarSensor(**(sonar_params or default_sonar))

        if current_params:
            self.current_enabled = True
            self.cur_strength = current_params['strength']
            self.cur_period = current_params['period']
            self.cur_direction = current_params['direction']
        else:
            self.current_enabled = False

        self.goal_radius = (goal_params or {}).get('radius', 0.5)
        self.goal_color = (255, 255, 0)

        if beacon_params:
            self.beacon_enabled = True
            self.ping_interval = beacon_params.get('ping_interval', 1.0)
            self.pulse_duration = beacon_params.get('pulse_duration', 0.1)
            self.beacon_intensity = beacon_params.get('beacon_intensity', 1.0)
            self.ping_noise = beacon_params.get('ping_noise', 0.01)
        else:
            self.beacon_enabled = False

        self.reset()

    def _build_maps(self):
        H, W = self.grid_size
        self.occ_grid = np.zeros((H, W), dtype=np.uint8)
        self.refl_grid = np.full((H, W), 0.2)
        for _ in range(50):
            r = np.random.randint(3, 10)
            cx = np.random.randint(r, W - r)
            cy = np.random.randint(r, H - r)
            xs = np.arange(cx - r, cx + r + 1)
            ys = np.arange(cy - r, cy + r + 1)
            xv, yv = np.meshgrid(xs, ys)
            mask = (xv - cx)**2 + (yv - cy)**2 <= r**2
            self.occ_grid[yv[mask], xv[mask]] = 1
            self.refl_grid[yv[mask], xv[mask]] = np.random.uniform(0.5, 1.0)

    def reset(self):
        H, W = self.grid_size
        self.pose = np.array([W/2*self.resolution,
                              H/2*self.resolution, 0.0])
        free = np.argwhere(self.occ_grid == 0)
        dists = np.linalg.norm((free - np.array([H/2, W/2])) * self.resolution, axis=1)
        min_d = min(H, W) * self.resolution * 0.3
        candidates = free[dists > min_d] if len(free) else free
        idx = candidates[np.random.randint(len(candidates))]
        self.goal = np.array([idx[1]*self.resolution, idx[0]*self.resolution])
        self.time = 0.0
        return self._get_obs()

    def _get_obs(self):
        ranges, intensities, hit_mask = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose)
        if self.beacon_enabled and (self.time % self.ping_interval) < self.pulse_duration:
            dx, dy = self.goal - self.pose[:2]
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx) - self.pose[2]
            bearing = (bearing + math.pi) % (2*math.pi) - math.pi
            idx = np.argmin(np.abs(self.sonar.beam_angles - bearing))
            if dist < self.sonar.max_range:
                ranges[idx] = dist + np.random.normal(0, self.ping_noise)
                hit_mask[idx] = True
                if intensities is not None:
                    intensities[idx] = self.beacon_intensity
        return ranges, intensities, hit_mask

    def step(self, action):
        v, omega = action
        dt = 0.1
        self.pose[2] += omega * dt
        self.pose[0] += v * dt * math.cos(self.pose[2])
        self.pose[1] += v * dt * math.sin(self.pose[2])
        if self.current_enabled:
            mag = self.cur_strength * math.sin(2*math.pi*self.time / self.cur_period)
            self.pose[0] += mag * math.cos(self.cur_direction) * dt
            self.pose[1] += mag * math.sin(self.cur_direction) * dt
        self.time += dt

        gi = int(self.pose[1] / self.resolution)
        gj = int(self.pose[0] / self.resolution)
        done = False
        reward = -0.01
        if (gi < 0 or gi >= self.occ_grid.shape[0] or
            gj < 0 or gj >= self.occ_grid.shape[1] or
            self.occ_grid[gi, gj] == 1):
            done, reward = True, -1.0
            obs = None
        elif np.linalg.norm(self.pose[:2] - self.goal) <= self.goal_radius:
            done, reward = True, +1.0
            obs = None
        else:
            obs = self._get_obs()

        return obs, reward, done, {}

    def render(self):
        surf = pygame.display.set_mode(self.window_size)
        surf.fill((0, 0, 50))
        map_w = 600
        H, W = self.grid_size
        cw = map_w / W
        ch = 600 / H

        for y, x in zip(*np.where(self.occ_grid)):
            pygame.draw.rect(surf, (100, 100, 100), (x*cw, y*ch, cw, ch))
        gx_pix = self.goal[0]/self.resolution * cw
        gy_pix = self.goal[1]/self.resolution * ch
        pygame.draw.circle(surf, self.goal_color,
                           (int(gx_pix), int(gy_pix)),
                           int(self.goal_radius/self.resolution * cw), 2)
        x_pix = self.pose[0]/self.resolution * cw
        y_pix = self.pose[1]/self.resolution * ch
        pygame.draw.circle(surf, (0,255,0), (int(x_pix), int(y_pix)), 5)

        # get obs including beacon
        ranges, intensities, hit_mask = self._get_obs()
        # draw beams
        for r, rel_ang, hit in zip(ranges, self.sonar.beam_angles, hit_mask):
            if not hit: continue
            ang = self.pose[2] + rel_ang
            x2 = x_pix + (r/self.resolution)*cw * math.cos(ang)
            y2 = y_pix + (r/self.resolution)*ch * math.sin(ang)
            pygame.draw.line(surf, (0,200,200), (x_pix,y_pix), (x2,y2), 1)

        # sonar panel
        sx0 = map_w
        sw = self.window_size[0] - map_w
        pygame.draw.rect(surf, (20,20,80), (sx0,0,sw,600))
        bs = sw / self.sonar.n_beams
        for i, (r, inten, hit) in enumerate(zip(ranges, intensities if intensities is not None else [None]*len(ranges), hit_mask)):
            if not hit: continue
            px = sx0 + i*bs + bs/2
            py = 600 - (r/self.sonar.max_range)*580
            if inten is not None:
                norm = min(max(inten, 0.0), 1.0)
                color = (int(norm*255), 0, int((1-norm)*255))
            else:
                color = (0,200,200)
            pygame.draw.circle(surf, color, (int(px), int(py)), 6)  # larger marker

        for i, r, inten in self.sonar.get_spurious(ranges, intensities, hit_mask):
            if r >= self.sonar.max_range: continue
            px = sx0 + i*bs + bs/2
            py = 600 - (r/self.sonar.max_range)*580
            norm = min(max(inten or 0, 0.0), 1.0)
            col = (150,150,150) if norm < 0.05 else (int(norm*255), 255-int(norm*255), 0)
            pygame.draw.circle(surf, col, (int(px), int(py)), 2)

        # Highlight beacon ping
        if self.beacon_enabled and (self.time % self.ping_interval) < self.pulse_duration:
            dx, dy = self.goal - self.pose[:2]
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx) - self.pose[2]
            bearing = (bearing + math.pi) % (2*math.pi) - math.pi
            idx = np.argmin(np.abs(self.sonar.beam_angles - bearing))
            if dist < self.sonar.max_range:
                px = sx0 + idx*bs + bs/2
                py = 600 - (dist/self.sonar.max_range)*580
                pygame.draw.circle(surf, (255,0,0), (int(px), int(py)), 10, 2)  # bold red ring

        pygame.display.flip()


def main():
    pygame.init()
    env = AUVEnv(
        sonar_params    = {'compute_intensity': True},
        current_params  = {'strength': 0.2, 'period': 30.0, 'direction': np.deg2rad(45)},
        goal_params     = {'radius': 0.5},
        beacon_params   = {'ping_interval': 1.0, 'pulse_duration': 0.1,
                            'beacon_intensity': 1.0, 'ping_noise': 0.01}
    )
    clock = pygame.time.Clock()
    running = True
    obs = env.reset()
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        v     =  1.0 if keys[pygame.K_UP]   else -1.0 if keys[pygame.K_DOWN]  else 0.0
        omega =  1.0 if keys[pygame.K_LEFT] else -1.0 if keys[pygame.K_RIGHT] else 0.0
        obs, rew, done, _ = env.step((v, omega))
        if done:
            obs = env.reset()
        env.render()
        clock.tick(60)
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()