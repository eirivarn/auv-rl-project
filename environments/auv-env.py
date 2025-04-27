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
                # Exiting map: ignore beam
                if gi < 0 or gi >= H or gj < 0 or gj >= W:
                    break
                # Obstacle hit
                if occ_grid[gi, gj]:
                    ranges[i] = r
                    hit_mask[i] = True
                    if self.compute_intensity:
                        base = refl_grid[gi, gj]
                        loss = r**2 if self.spreading_loss and r > 0 else 1.0
                        intensities[i] = base / loss
                    break

        # Add noise only for valid hits
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
        # Debris echoes
        for _ in range(np.random.poisson(self.debris_rate)):
            i = np.random.randint(0, self.n_beams)
            r = np.random.uniform(0, self.max_range)
            inten = np.random.uniform(0, 0.1)
            spurious.append((i, r, inten))
        # Ghost echoes (multipath)
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
                 window_size=(800, 600)):
        self.grid_size = grid_size
        self.resolution = resolution
        self.window_size = window_size
        self._build_maps()

        default_sonar = {
            'fov': np.deg2rad(90),
            'n_beams': 60,
            'max_range': 10.0,
            'resolution': resolution,
            'noise_std': 0.01,
            'compute_intensity': False,
            'spreading_loss': True,
            'debris_rate': 5,
            'ghost_prob': 0.05,
            'ghost_decay': 0.3
        }
        params = sonar_params or default_sonar
        self.sonar = SonarSensor(**params)

        # Ocean current settings
        if current_params:
            self.current_enabled = True
            self.cur_strength = current_params.get('strength', 0.1)
            self.cur_period = current_params.get('period', 20.0)
            self.cur_direction = current_params.get('direction', 0.0)
            self.time = 0.0
        else:
            self.current_enabled = False

        self.reset()

    def _build_maps(self):
        H, W = self.grid_size
        self.occ_grid = np.zeros((H, W), dtype=np.uint8)
        self.refl_grid = np.full((H, W), 0.2)
        # Random circular obstacles
        for _ in range(50):
            radius = np.random.randint(3, 10)
            cx = np.random.randint(radius, W - radius)
            cy = np.random.randint(radius, H - radius)
            xs = np.arange(cx-radius, cx+radius+1)
            ys = np.arange(cy-radius, cy+radius+1)
            xv, yv = np.meshgrid(xs, ys)
            mask = (xv-cx)**2 + (yv-cy)**2 <= radius**2
            self.occ_grid[yv[mask], xv[mask]] = 1
            self.refl_grid[yv[mask], xv[mask]] = np.random.uniform(0.5, 1.0)

    def reset(self):
        H, W = self.grid_size
        self.pose = np.array([W/2*self.resolution,
                              H/2*self.resolution,
                              0.0])
        self.time = 0.0
        return self._get_obs()

    def _get_obs(self):
        return self.sonar.get_readings(self.occ_grid, self.refl_grid, self.pose)

    def step(self, action):
        v, omega = action
        dt = 0.1
        # Control
        self.pose[2] += omega * dt
        self.pose[0] += v * dt * math.cos(self.pose[2])
        self.pose[1] += v * dt * math.sin(self.pose[2])
        # Ocean current
        if self.current_enabled:
            mag = self.cur_strength * math.sin(2*math.pi*self.time / self.cur_period)
            dx = mag * math.cos(self.cur_direction) * dt
            dy = mag * math.sin(self.cur_direction) * dt
            self.pose[0] += dx
            self.pose[1] += dy
            self.time += dt
        return self._get_obs(), 0.0, False, {}

    def render(self):
        surf = pygame.display.set_mode(self.window_size)
        surf.fill((0, 0, 50))
        map_w = 600
        H, W = self.grid_size
        cw = map_w / W
        ch = 600 / H

        # Draw map and robot
        for y, x in zip(*np.where(self.occ_grid)):
            pygame.draw.rect(surf, (100,100,100), (x*cw, y*ch, cw, ch))
        x_pix = self.pose[0]/self.resolution * cw
        y_pix = self.pose[1]/self.resolution * ch
        pygame.draw.circle(surf, (0,255,0), (int(x_pix), int(y_pix)), 5)

        # Sensor readings
        ranges, intensities, hit_mask = self.sonar.get_readings(self.occ_grid, self.refl_grid, self.pose)

        # Draw primary beams (hits only)
        for r, rel_ang, hit in zip(ranges, self.sonar.beam_angles, hit_mask):
            if not hit: continue
            ang = self.pose[2] + rel_ang
            x2 = x_pix + (r/self.resolution)*cw * math.cos(ang)
            y2 = y_pix + (r/self.resolution)*ch * math.sin(ang)
            pygame.draw.line(surf, (0,200,200), (x_pix,y_pix), (x2,y2), 1)

        # Sonar display panel
        sonar_x0 = map_w
        sonar_w = self.window_size[0] - map_w
        pygame.draw.rect(surf, (20,20,80), (sonar_x0,0,sonar_w,600))
        beam_spacing = sonar_w / self.sonar.n_beams

        # Draw primary returns
        for i, (r, inten, hit) in enumerate(zip(ranges, intensities if intensities is not None else [None]*len(ranges), hit_mask)):
            if not hit: continue
            px = sonar_x0 + i*beam_spacing + beam_spacing/2
            py = 600 - (r/self.sonar.max_range)*580
            if inten is not None:
                norm = min(max(inten, 0.0), 1.0)
                color = (int(norm*255), 0, int((1-norm)*255))
            else:
                color = (0,200,200)
            pygame.draw.circle(surf, color, (int(px), int(py)), 3)

        # Draw spurious echoes
        for i, r, inten in self.sonar.get_spurious(ranges, intensities, hit_mask):
            if r >= self.sonar.max_range: continue
            px = sonar_x0 + i*beam_spacing + beam_spacing/2
            py = 600 - (r/self.sonar.max_range)*580
            norm = min(max(inten or 0, 0.0), 1.0)
            col = (150,150,150) if norm < 0.05 else (int(norm*255), 255-int(norm*255), 0)
            pygame.draw.circle(surf, col, (int(px), int(py)), 2)

        pygame.display.flip()


def main():
    pygame.init()
    env = AUVEnv(
        sonar_params={'compute_intensity': True},
        current_params={'strength': 0.2, 'period': 30.0, 'direction': np.deg2rad(45)}
    )
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        v = omega = 0.0
        if keys[pygame.K_UP]:    v = 1.0
        if keys[pygame.K_DOWN]:  v = -1.0
        if keys[pygame.K_LEFT]:  omega = 1.0
        if keys[pygame.K_RIGHT]: omega = -1.0
        env.step((v, omega))
        env.render()
        clock.tick(30)
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
