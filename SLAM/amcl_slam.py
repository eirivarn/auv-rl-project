import numpy as np
import pygame
import sys
import math
from environments.simple_env import AUVEnv

# SLAM & docking parameters
LOG_ODDS_FREE = np.log(0.3/0.7)
LOG_ODDS_OCC  = np.log(0.7/0.3)
MIN_LOG_ODDS = -5.0
MAX_LOG_ODDS = 5.0
FOV = 2 * math.pi
BEAMS = 60
MAX_RANGE = 10.0
RESOLUTION = 0.05
ANGLE_TOL = 0.5  # radians
FORWARD_SPEED = 0.2
ROT_SPEED = 1.0

# Alternate turn flag
toggle_left = True


def update_map(log_odds, pose, ranges, hit_mask, beam_angles, iter_num):
    x, y, theta = pose
    print(f"Iter {iter_num}: hits={hit_mask.sum()}, ranges min={ranges.min():.2f}, max={ranges.max():.2f}")
    for i, r in enumerate(ranges):
        if not hit_mask[i]:
            continue
        ang = theta + beam_angles[i]
        steps = int(min(r, MAX_RANGE) / RESOLUTION)
        for s in range(steps):
            d = (s + 1) * RESOLUTION
            xi = x + d * math.cos(ang)
            yi = y + d * math.sin(ang)
            gi, gj = int(yi / RESOLUTION), int(xi / RESOLUTION)
            if 0 <= gi < log_odds.shape[0] and 0 <= gj < log_odds.shape[1]:
                log_odds[gi, gj] += LOG_ODDS_FREE
        xi = x + r * math.cos(ang)
        yi = y + r * math.sin(ang)
        gi, gj = int(yi / RESOLUTION), int(xi / RESOLUTION)
        if 0 <= gi < log_odds.shape[0] and 0 <= gj < log_odds.shape[1]:
            log_odds[gi, gj] += LOG_ODDS_OCC
    np.clip(log_odds, MIN_LOG_ODDS, MAX_LOG_ODDS, out=log_odds)
    print(f"Map update: log-odds min={log_odds.min():.2f}, max={log_odds.max():.2f}")
    return log_odds


def is_clear_path(env, start, goal):
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    dist = math.hypot(dx, dy)
    steps = int(dist / RESOLUTION)
    for i in range(1, steps+1):
        t = i/steps
        x = start[0] + t * dx
        y = start[1] + t * dy
        gi, gj = int(y/RESOLUTION), int(x/RESOLUTION)
        if 0 <= gi < env.occ_grid.shape[0] and 0 <= gj < env.occ_grid.shape[1]:
            if env.occ_grid[gi, gj]:
                return False
    return True


def main():
    global toggle_left
    pygame.init()
    screen = pygame.display.set_mode((1200, 600))
    clock = pygame.time.Clock()

    env = AUVEnv(
        sonar_params={
            'fov': FOV, 'n_beams': BEAMS,
            'max_range': MAX_RANGE, 'resolution': RESOLUTION,
            'compute_intensity': True
        },
        beacon_params={
            'ping_interval': 1.0,
            'pulse_duration': 0.1,
            'beacon_intensity': 1.0,
            'ping_noise': 0.01
        },
        current_params=None,
        goal_params=None,
        window_size=(600,600)
    )

    H, W = env.grid_size
    log_odds = np.zeros((H, W), dtype=float)
    beam_angles = env.sonar.beam_angles
    state = 'go_to_goal'
    hit_point = None
    iter_num = 0

    obs = env.reset()

    while True:
        # handle events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # SLAM map update
        ranges, intensities, hit_mask = obs
        log_odds = update_map(log_odds, env.pose, ranges, hit_mask, beam_angles, iter_num)

        # compute goal metrics
        dx = env.goal[0] - env.pose[0]
        dy = env.goal[1] - env.pose[1]
        dist_goal = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        rel_bearing = (angle_to_goal - env.pose[2] + math.pi) % (2*math.pi) - math.pi
        front_idx = np.where(np.abs(env.sonar.beam_angles) < math.pi/12)[0]
        min_front = np.min(ranges[front_idx]) if front_idx.size else MAX_RANGE
        print(f"State={state}, dist={dist_goal:.2f}, rel_bearing={rel_bearing:.2f}, min_front={min_front:.2f}")

        # BUG2/go_to_goal + alternate wall turns
        if dist_goal <= env.goal_radius:
            v, omega = 0.0, 0.0
            print("Reached docking station!")
        elif state == 'go_to_goal':
            if is_clear_path(env, env.pose[:2], env.goal):
                # clear sight: go straight
                v, omega = FORWARD_SPEED, 0.0
                print("Clear path: moving to goal")
            elif min_front < 1.0:
                # hit obstacle: alternate turn direction
                v = 0.0
                omega = ROT_SPEED if toggle_left else -ROT_SPEED
                toggle_left = not toggle_left
                state = 'wall_follow'
                print(f"Hit obstacle: alternating turn, omega={omega:.2f}")
            else:
                # heading correction toward goal
                v = FORWARD_SPEED
                omega = math.copysign(ROT_SPEED/4, rel_bearing)
        elif state == 'wall_follow':
            if is_clear_path(env, env.pose[:2], env.goal):
                # regain clear sight: resume go_to_goal
                state = 'go_to_goal'
                v, omega = FORWARD_SPEED, 0.0
                print("Sight regained: switch to go_to_goal")
            elif min_front < 1.0:
                # continue wall follow: alternate turns
                v = 0.0
                omega = ROT_SPEED if toggle_left else -ROT_SPEED
                toggle_left = not toggle_left
                print(f"Wall follow: alternating turn, omega={omega:.2f}")
            else:
                # move forward along wall
                v = FORWARD_SPEED
                omega = 0.0
        else:
            v, omega = 0.0, 0.0

        print(f"Control: v={v:.2f}, omega={omega:.2f}")

        # step environment
        obs, _, done, _ = env.step((v, omega))
        if done:
            obs = env.reset()
            state = 'go_to_goal'
            hit_point = None
        iter_num += 1

        # --- Visualization (two panels) ---
        screen.fill((0, 0, 0))
        total_w, total_h = screen.get_size()
        left_w = total_w // 2
        right_w = total_w - left_w
        # cell sizes
        cwL = left_w / W
        chL = total_h / H
        cwR = right_w / W
        chR = total_h / H

        # Left panel: true environment
        for i, j in zip(*np.where(env.occ_grid)):
            pygame.draw.rect(screen, (100,100,100), (j*cwL, i*chL, cwL, chL))
        # robot pose
        xpix = env.pose[0]/RESOLUTION * cwL
        ypix = env.pose[1]/RESOLUTION * chL
        pygame.draw.circle(screen, (0,255,0), (int(xpix), int(ypix)), 5)
        # sonar beams
        for r, rel, hit in zip(ranges, env.sonar.beam_angles, hit_mask):
            ang = env.pose[2] + rel
            x2 = xpix + (r/RESOLUTION)*cwL * math.cos(ang)
            y2 = ypix + (r/RESOLUTION)*chL * math.sin(ang)
            pygame.draw.line(screen, (0,200,200), (xpix, ypix), (x2, y2), 1)
            if hit:
                pygame.draw.circle(screen, (255,0,0), (int(x2), int(y2)), 3)

        # Right panel: SLAM occupancy map
        offset = left_w
        for gi in range(H):
            for gj in range(W):
                if log_odds[gi, gj] > 0:
                    pygame.draw.rect(screen, (200,200,200),
                                     (offset + gj*cwR, gi*chR, cwR, chR))
        # docking station goal (cyan)
        gx = offset + env.goal[0]/RESOLUTION * cwR
        gy = env.goal[1]/RESOLUTION * chR
        pygame.draw.circle(screen, (0,255,255), (int(gx), int(gy)), 8, 2)
        # estimated pose (blue)
        ex = offset + env.pose[0]/RESOLUTION * cwR
        ey = env.pose[1]/RESOLUTION * chR
        pygame.draw.circle(screen, (0,0,255), (int(ex), int(ey)), 5)

        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    main()
