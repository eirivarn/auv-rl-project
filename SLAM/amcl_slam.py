import numpy as np
import pygame
import sys
import math
from environments.test_env import AUVEnv  # adjust import path as needed

class ParticleFilter:
    """
    Particle Filter using range-based AMCL update.
    """
    def __init__(self, env, n_particles=200, motion_noise=(0.01,0.005), sensor_noise=0.1):
        self.env = env
        self.n = n_particles
        self.motion_noise = motion_noise
        self.sensor_noise = sensor_noise

        # initialize particles uniformly in free space (meters)
        free = np.argwhere(env.occ_grid == 0)
        coords = free * env.resolution  # (row,col)->(y,x)
        idx = np.random.choice(len(coords), size=self.n)
        pts = coords[idx]
        thetas = np.random.uniform(-math.pi, math.pi, size=self.n)
        self.p = np.column_stack((pts[:,1], pts[:,0], thetas))  # (x, y, theta)
        self.w = np.ones(self.n) / self.n

    def predict(self, v, omega, dt=0.1):
        v_noisy = v + np.random.normal(0, self.motion_noise[0], self.n)
        o_noisy = omega + np.random.normal(0, self.motion_noise[1], self.n)
        th = (self.p[:,2] + o_noisy * dt) % (2*math.pi)
        x = self.p[:,0] + v_noisy * dt * np.cos(th)
        y = self.p[:,1] + v_noisy * dt * np.sin(th)
        self.p = np.column_stack((x, y, th))

    def update(self, obs_ranges, obs_mask):
        sigma = self.sensor_noise
        beams = np.where(obs_mask)[0]
        if len(beams) == 0:
            self.w.fill(1.0/self.n)
            return
        sel = beams[::max(1, len(beams)//15)]

        log_w = np.zeros(self.n)
        for i in range(self.n):
            x, y, theta = self.p[i]
            sim_ranges, _, _ = self.env.sonar.get_readings(
                self.env.occ_grid, self.env.refl_grid, (x, y, theta)
            )
            diffs = obs_ranges[sel] - sim_ranges[sel]
            log_w[i] = -0.5 * np.sum((diffs / sigma)**2)
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        self.w = w / np.sum(w)

    def resample(self):
        positions = (np.arange(self.n) + np.random.rand()) / self.n
        cdf = np.cumsum(self.w)
        idx = np.searchsorted(cdf, positions)
        self.p = self.p[idx]
        self.w.fill(1.0/self.n)

    def estimate(self):
        x = np.average(self.p[:,0], weights=self.w)
        y = np.average(self.p[:,1], weights=self.w)
        sin_t = np.average(np.sin(self.p[:,2]), weights=self.w)
        cos_t = np.average(np.cos(self.p[:,2]), weights=self.w)
        theta = math.atan2(sin_t, cos_t)
        return x, y, theta

# ---------------------------------------------------
# Main: simulation and PF on separate panels
# ---------------------------------------------------
if __name__=='__main__':
    pygame.init()
    # two 600x600 panels side by side
    screen = pygame.display.set_mode((1200, 600))
    clock = pygame.time.Clock()

    env = AUVEnv(
        sonar_params={'compute_intensity': False},
        current_params=None,
        goal_params=None,
        beacon_params=None,
        window_size=(600,600)
    )
    pf = ParticleFilter(env)
    obs = env.reset()
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        v =  1.0 if keys[pygame.K_UP] else -1.0 if keys[pygame.K_DOWN] else 0.0
        omega =  1.0 if keys[pygame.K_LEFT] else -1.0 if keys[pygame.K_RIGHT] else 0.0

        # true step
        obs, _, done, _ = env.step((v, omega))
        if done:
            obs = env.reset()
            pf = ParticleFilter(env)
        obs_ranges, _, hit_mask = obs

        # PF update
        pf.predict(v, omega)
        pf.update(obs_ranges, hit_mask)
        pf.resample()
        ex, ey, et = pf.estimate()

        # clear screen
        screen.fill((0,0,0))
        map_w = 600
        H, W = env.grid_size
        cw = map_w / W
        ch = map_w / H

        # --- left panel: simulation world ---
        # draw obstacles
        for y, x in zip(*np.where(env.occ_grid)):
            pygame.draw.rect(screen, (100,100,100), (x*cw, y*ch, cw, ch))
        # draw robot and its beams
        x_pix = env.pose[0]/env.resolution * cw
        y_pix = env.pose[1]/env.resolution * ch
        pygame.draw.circle(screen, (0,255,0), (int(x_pix),int(y_pix)), 5)
        for r, rel_ang, hit in zip(obs_ranges, env.sonar.beam_angles, hit_mask):
            ang = env.pose[2] + rel_ang
            x2 = x_pix + (r/env.resolution)*cw * math.cos(ang)
            y2 = y_pix + (r/env.resolution)*ch * math.sin(ang)
            pygame.draw.line(screen, (0,200,200), (x_pix,y_pix), (x2,y2), 1)
            if hit:
                pygame.draw.circle(screen, (255,0,0), (int(x2),int(y2)), 3)

        # --- right panel: PF map ---
        offset = map_w
        # draw map cells (same ordering as left)
        for y, x in zip(*np.where(env.occ_grid)):
            pygame.draw.rect(screen, (50,50,50), (offset + x*cw, y*ch, cw, ch))

        # transform the exact observed ranges (obs_ranges, hit_mask) into world coords
        angles = env.sonar.beam_angles + env.pose[2]
        xs = obs_ranges * np.cos(angles)
        ys = obs_ranges * np.sin(angles)
        world_pts = np.stack((env.pose[0] + xs, env.pose[1] + ys), axis=1)

        # draw sonar hits in world coords
        for (wx, wy), m in zip(world_pts, hit_mask):
            if not m: continue
            px = offset + (wx/env.resolution) * cw
            py = (wy/env.resolution) * ch
            pygame.draw.circle(screen, (255,0,0), (int(px), int(py)), 4)
        # draw particles
        for x_p, y_p, _ in pf.p:
            px = offset + (x_p/env.resolution) * cw
            py = (y_p/env.resolution) * ch
            pygame.draw.circle(screen, (0,0,255), (int(px), int(py)), 2)
        # draw estimate
        px = offset + (ex/env.resolution) * cw
        py = (ey/env.resolution) * ch
        pygame.draw.circle(screen, (255,255,0), (int(px), int(py)), 6, 2)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()
