import numpy as np
import pygame
import sys
import math
from environments.test_env import AUVEnv  

# Particle filter parameters
NUM_PARTICLES = 50
MOTION_NOISE = {'v': 0.1, 'omega': 0.05}
SENSOR_NOISE = 0.1
BEAMS = 10  # number of beams for subsampling

# Initialize particles uniformly over free map cells
def init_particles(env):
    free = np.argwhere(env.occ_grid == 0)
    indices = np.random.choice(len(free), size=NUM_PARTICLES, replace=True)
    particles = []
    for gi, gj in free[indices]:
        x = (gj + 0.5) * env.resolution
        y = (gi + 0.5) * env.resolution
        theta = np.random.uniform(-math.pi, math.pi)
        particles.append([x, y, theta])
    return np.array(particles)

# Motion update with noise
def motion_update(particles, control, dt=0.1):
    v, omega = control
    noisy_v = v + np.random.normal(0, MOTION_NOISE['v'], size=len(particles))
    noisy_o = omega + np.random.normal(0, MOTION_NOISE['omega'], size=len(particles))
    particles[:,2] += noisy_o * dt
    particles[:,0] += noisy_v * dt * np.cos(particles[:,2])
    particles[:,1] += noisy_v * dt * np.sin(particles[:,2])
    return particles

# Expected ranges by ray-casting for given beam angles
def expected_ranges(pose, env, beam_angles):
    x, y, theta = pose
    exp = np.full(len(beam_angles), env.sonar.max_range)
    res = env.resolution
    for i, rel in enumerate(beam_angles):
        ang = theta + rel
        for r in np.arange(0, env.sonar.max_range, res):
            xi = x + r * math.cos(ang)
            yi = y + r * math.sin(ang)
            gi = int(yi / res)
            gj = int(xi / res)
            if gi < 0 or gi >= env.occ_grid.shape[0] or gj < 0 or gj >= env.occ_grid.shape[1]:
                exp[i] = r
                break
            if env.occ_grid[gi, gj]:
                exp[i] = r
                break
    return exp

# Sensor update: compute log-likelihood and normalize to avoid underflow
def sensor_update(particles, weights, obs_ranges, beam_angles, hit_mask, env):
    N = len(particles)
    log_w = np.zeros(N)
    for i, p in enumerate(particles):
        exp = expected_ranges(p, env, beam_angles)
        # hit beams
        mask = hit_mask.astype(bool)
        if np.any(mask):
            diff_hit = obs_ranges[mask] - exp[mask]
            log_w[i] = -0.5 * np.sum((diff_hit / SENSOR_NOISE)**2)
        # missed beams: encourage exp to be near max_range
        miss = ~mask
        if np.any(miss):
            diff_miss = env.sonar.max_range - exp[miss]
            log_w[i] += -0.5 * np.sum((diff_miss / SENSOR_NOISE)**2)
    # normalize log weights to prevent underflow
    max_log = np.max(log_w)
    w = np.exp(log_w - max_log)
    w /= np.sum(w)
    return w


# Systematic resampling
def resample(particles, weights):
    N = len(particles)
    positions = (np.arange(N) + np.random.rand()) / N
    cum = np.cumsum(weights)
    idx = np.searchsorted(cum, positions)
    return particles[idx].copy(), np.ones(N) / N

# Estimate pose by weighted average
def estimate_pose(particles, weights):
    x = np.average(particles[:,0], weights=weights)
    y = np.average(particles[:,1], weights=weights)
    cos_s = np.average(np.cos(particles[:,2]), weights=weights)
    sin_s = np.average(np.sin(particles[:,2]), weights=weights)
    theta = math.atan2(sin_s, cos_s)
    return np.array([x, y, theta])

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((1200, 600))
    clock = pygame.time.Clock()

    env = AUVEnv(
        sonar_params={'compute_intensity': False},
        current_params=None, goal_params=None, beacon_params=None,
        window_size=(600,600)
    )
    obs = env.reset()

    # Precompute beam indices and angles
    total = env.sonar.n_beams
    beam_idxs = np.linspace(0, total-1, BEAMS, dtype=int)
    beam_angles = env.sonar.beam_angles[beam_idxs]

    # Initialize PF
    particles = init_particles(env)
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

    iteration = 0
    while True:
        # Handle events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Control input
        keys = pygame.key.get_pressed()
        v = 1.0 if keys[pygame.K_UP] else -1.0 if keys[pygame.K_DOWN] else 0.0
        omega = 1.0 if keys[pygame.K_LEFT] else -1.0 if keys[pygame.K_RIGHT] else 0.0

        # True step
        obs, _, done, _ = env.step((v, omega))
        if done:
            obs = env.reset()
            particles = init_particles(env)
            weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

        full_ranges, _, full_hit = obs
        obs_ranges = full_ranges[beam_idxs]
        hit_mask = full_hit[beam_idxs]

        # PF predict
        particles = motion_update(particles, (v, omega))
        # PF update
        weights = sensor_update(particles, weights, obs_ranges, beam_angles, hit_mask, env)
        # Resample
        particles, weights = resample(particles, weights)
        # Estimate
        est = estimate_pose(particles, weights)

        # Render panels
        screen.fill((0,0,0))
        total_w, total_h = screen.get_size()
        map_w = total_w // 2
        H, W = env.grid_size
        cw = map_w / W
        ch = total_h / H

        # Left panel: simulation
        for y, x in zip(*np.where(env.occ_grid)):
            pygame.draw.rect(screen, (100,100,100), (x*cw, y*ch, cw, ch))
        xpix = env.pose[0]/env.resolution * cw
        ypix = env.pose[1]/env.resolution * ch
        pygame.draw.circle(screen, (0,255,0), (int(xpix), int(ypix)), 5)
        for r, rel, hit in zip(full_ranges, env.sonar.beam_angles, full_hit):
            ang = env.pose[2] + rel
            x2 = xpix + (r/env.resolution)*cw * math.cos(ang)
            y2 = ypix + (r/env.resolution)*ch * math.sin(ang)
            pygame.draw.line(screen, (0,200,200), (xpix, ypix), (x2, y2), 1)
            if hit:
                pygame.draw.circle(screen, (255,0,0), (int(x2), int(y2)), 3)

        # Right panel: PF map
        offset = map_w
        # draw map background
        for y, x in zip(*np.where(env.occ_grid)):
            pygame.draw.rect(screen, (50,50,50), (offset + x*cw, y*ch, cw, ch))
        # draw sonar hits
        angles_full = env.sonar.beam_angles + env.pose[2]
        xs = full_ranges * np.cos(angles_full)
        ys = full_ranges * np.sin(angles_full)
        wpts = np.stack((env.pose[0] + xs, env.pose[1] + ys), axis=1)
        for (wx, wy), m in zip(wpts, full_hit):
            if not m: continue
            px = offset + (wx/env.resolution) * cw
            py = (wy/env.resolution) * ch
            pygame.draw.circle(screen, (255,0,0), (int(px), int(py)), 4)

        # draw particles
        for x, y, theta in particles:
            px = offset + (x/env.resolution) * cw
            py = (y/env.resolution) * ch
            pygame.draw.circle(screen, (255,0,0), (int(px), int(py)), 2)
            dx = math.cos(theta) * cw
            dy = math.sin(theta) * ch
            pygame.draw.line(screen, (255,0,0), (int(px), int(py)), (int(px+dx), int(py+dy)), 1)

        # draw true pose (green) on right
        px_true = offset + (env.pose[0] / env.resolution) * cw
        py_true = (env.pose[1] / env.resolution) * ch
        pygame.draw.circle(screen, (0,255,0), (int(px_true), int(py_true)), 5)
        dx_t = math.cos(env.pose[2]) * cw
        dy_t = math.sin(env.pose[2]) * ch
        pygame.draw.line(screen, (0,255,0), (int(px_true), int(py_true)), (int(px_true+dx_t), int(py_true+dy_t)), 2)

        # draw estimated pose (blue) on right
        px_est = offset + (est[0] / env.resolution) * cw
        py_est = (est[1] / env.resolution) * ch
        pygame.draw.circle(screen, (0,0,255), (int(px_est), int(py_est)), 5)
        dx_e = math.cos(est[2]) * cw
        dy_e = math.sin(est[2]) * ch
        pygame.draw.line(screen, (0,0,255), (int(px_est), int(py_est)), (int(px_est+dx_e), int(py_est+dy_e)), 2)

        pygame.display.flip()
        clock.tick(30)
        iteration += 1
