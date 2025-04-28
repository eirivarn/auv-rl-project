import numpy as np
import pygame
import sys
import math
import heapq
from scipy.ndimage import binary_dilation
from environments.test_env import AUVEnv

# --- A* on a 2D grid ---
def astar(grid, start, goal):
    h = lambda a,b: abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = [(h(start,goal), 0, start, None)]
    came_from = {}
    cost_so_far = {start: 0}
    while open_set:
        f, g, current, parent = heapq.heappop(open_set)
        if current in came_from:
            continue
        came_from[current] = parent
        if current == goal:
            break
        for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
            nb = (current[0]+di, current[1]+dj)
            if not (0 <= nb[0] < grid.shape[0] and 0 <= nb[1] < grid.shape[1]):
                continue
            if grid[nb]:
                continue
            new_cost = g + 1
            if new_cost < cost_so_far.get(nb, np.inf):
                cost_so_far[nb] = new_cost
                heapq.heappush(open_set, (new_cost + h(nb,goal), new_cost, nb, current))
    if goal not in came_from:
        return None
    path = []
    node = goal
    while node:
        path.append(node)
        node = came_from[node]
    return path[::-1]

def to_grid(pose, res):
    # Convert world (x,y) to grid indices (i,j).
    x, y, _ = pose
    return int(y/res), int(x/res)

def to_world(cell, res):
    # Convert grid (i,j) to world (x,y) at cell center.
    return ((cell[1]+0.5)*res, (cell[0]+0.5)*res)

# Planner parameters
REPLAN_INTERVAL = 30
DIRECT_SPEED = 0.2
ROT_SPEED = 0.5
ANGLE_TOL = 0.1

if __name__=='__main__':
    pygame.init()
    screen = pygame.display.set_mode((1200, 600))
    clock = pygame.time.Clock()

    env = AUVEnv(
        sonar_params={'compute_intensity': False},
        current_params=None,
        goal_params={'radius': 0.5},
        beacon_params={'ping_interval':1.0, 'pulse_duration':0.1,
                       'beacon_intensity':1.0,'ping_noise':0.01},
        window_size=(600,600)
    )
    obs = env.reset()

    step_count = 0
    current_path = None
    path_idx = 0
    occ_inf = np.zeros(env.grid_size, dtype=np.uint8)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # Obtain last observation
        ranges, intensities, hits = obs

        # Beacon detection
        beacon_beams = [i for i,inten in enumerate(intensities or [])
                        if hits[i] and inten >= env.beacon_intensity*0.9]

        # Decide control
        if beacon_beams:
            # Direct ping steering
            idx = beacon_beams[0]
            beam_angle = env.sonar.beam_angles[idx]
            if abs(beam_angle) > ANGLE_TOL:
                v = 0.0
                omega = math.copysign(ROT_SPEED, beam_angle)
            else:
                v = DIRECT_SPEED
                omega = 0.0
        else:
            # Global A* planner on true map for robustness
            if step_count % REPLAN_INTERVAL == 0 or current_path is None:
                # Build occupancy & inflate
                occ = env.occ_grid.copy()
                occ_inf = binary_dilation(occ, structure=np.ones((3,3))).astype(np.uint8)
                start = to_grid(env.pose, env.resolution)
                goal_cell = (int(env.goal[1]/env.resolution),
                             int(env.goal[0]/env.resolution))
                current_path = astar(occ_inf, start, goal_cell)
                path_idx = 0

            if current_path and len(current_path) > 1:
                next_cell = current_path[min(path_idx+1, len(current_path)-1)]
                wx, wy = to_world(next_cell, env.resolution)
                dx = wx - env.pose[0]
                dy = wy - env.pose[1]
                desired = math.atan2(dy, dx)
                db = (desired - env.pose[2] + math.pi) % (2*math.pi) - math.pi
                if abs(db) > ANGLE_TOL:
                    v = 0.0
                    omega = math.copysign(ROT_SPEED, db)
                else:
                    v = DIRECT_SPEED
                    omega = 0.0
                    path_idx += 1
            else:
                # Fallback spin
                v = 0.0
                omega = ROT_SPEED

        # Step true environment
        obs, _, done, _ = env.step((v, omega))
        if done:
            obs = env.reset()

        # Draw left and right panels
        screen.fill((0,0,0))
        map_w = 600
        H, W = env.grid_size
        cw = map_w / W
        ch = map_w / H

        # Left: actual environment
        for y, x in zip(*np.where(env.occ_grid)):
            pygame.draw.rect(screen, (100,100,100), (x*cw, y*ch, cw, ch))
        x_pix = env.pose[0]/env.resolution * cw
        y_pix = env.pose[1]/env.resolution * ch
        pygame.draw.circle(screen, (0,255,0), (int(x_pix), int(y_pix)), 5)
        for r, rel_ang, hit in zip(ranges, env.sonar.beam_angles, hits):
            ang = env.pose[2] + rel_ang
            x2 = x_pix + (r/env.resolution)*cw * math.cos(ang)
            y2 = y_pix + (r/env.resolution)*ch * math.sin(ang)
            pygame.draw.line(screen, (0,200,200), (x_pix, y_pix), (x2, y2), 1)
            if hit:
                pygame.draw.circle(screen, (255,0,0), (int(x2), int(y2)), 3)

        # Right: planner path and inflated map
        offset = map_w
        for y, x in zip(*np.where(occ_inf)):
            pygame.draw.rect(screen, (50,50,50), (offset + x*cw, y*ch, cw, ch))
        if current_path:
            for cell in current_path:
                xw, yw = to_world(cell, env.resolution)
                px = offset + xw/env.resolution * cw
                py = yw/env.resolution * ch
                pygame.draw.circle(screen, (0,0,255), (int(px), int(py)), 3)
        gx = env.goal[0]/env.resolution * cw + offset
        gy = env.goal[1]/env.resolution * ch
        pygame.draw.circle(screen, (255,255,0), (int(gx), int(gy)), 6, 2)

        pygame.display.flip()
        clock.tick(30)
        step_count += 1
