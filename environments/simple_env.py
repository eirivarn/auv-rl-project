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
                 fov=np.deg2rad(360), n_beams=60,
                 max_range=10.0, resolution=0.05,
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
    def __init__(self, grid_size=(200, 200), resolution=0.05,
                 sonar_params=None, current_params=None,
                 goal_params=None, beacon_params=None,
                 window_size=(800, 600)):

        self.grid_size = grid_size
        self.resolution = resolution
        self.window_size = window_size
        self._build_maps()

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.previous_action = 0.0

        default_sonar = dict(
            fov=np.deg2rad(90), n_beams=60, max_range=10.0,
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

        self.goal_radius = (goal_params or {}).get('radius', 0.5)
        self.goal_color = (255, 255, 0)

        self.beacon_enabled = bool(beacon_params)
        if self.beacon_enabled:
            self.ping_interval = beacon_params['ping_interval']
            self.pulse_duration = beacon_params['pulse_duration']
            self.beacon_intensity = beacon_params.get('beacon_intensity', 1.0)
            self.ping_noise = beacon_params.get('ping_noise', 0.01)

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
    
  
    def _get_obs(self):
        ranges, intensities, hit_mask = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )

        heading = np.array([np.cos(self.pose[2]), np.sin(self.pose[2])])

        rel_goal_vec = self.goal - self.pose[:2]
        goal_distance = np.linalg.norm(rel_goal_vec)
        goal_angle = np.arctan2(rel_goal_vec[1], rel_goal_vec[0]) - self.pose[2]
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))

        obs = np.concatenate([
            ranges,
            intensities if intensities is not None else [],
            heading,
            [goal_distance, goal_angle],
            [self.linear_velocity, self.angular_velocity],
            [self.previous_action]
        ])
        return obs
     

    def _get_raw_obs(self):
        ranges, intensities, hit_mask = self.sonar.get_readings(
            self.occ_grid, self.refl_grid, self.pose
        )
        if self.beacon_enabled and (self.time % self.ping_interval) < self.pulse_duration:
            dx, dy = self.goal - self.pose[:2]
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx) - self.pose[2]
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            idx = np.argmin(np.abs(self.sonar.beam_angles - bearing))
            if dist < self.sonar.max_range:
                ranges[idx] = dist + np.random.normal(0, self.ping_noise)
                hit_mask[idx] = True
                if intensities is not None:
                    intensities[idx] = self.beacon_intensity
        return ranges, intensities, hit_mask

    def reset(self):
        H, W = self.grid_size

        x = W / 2 * self.resolution
        y = H / 2 * self.resolution
        theta = 0.0
        self.pose = np.array([x, y, theta])

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.previous_action = 0.0

        angle = np.random.uniform(-np.pi, np.pi)
        dist = np.random.uniform(2.0, 5.0)
        goal_x = x + dist * np.cos(angle)
        goal_y = y + dist * np.sin(angle)
        goal_x = np.clip(goal_x, 0, W * self.resolution)
        goal_y = np.clip(goal_y, 0, H * self.resolution)
        self.goal = np.array([goal_x, goal_y])

        dx, dy = self.goal - self.pose[:2]
        self.pose[2] = np.arctan2(dy, dx)

        self.time = 0.0
        return self._get_obs()

    def step(self, action):
        v, omega = action
        v = np.clip(v, -1.0, 1.0)
        omega = np.clip(omega, -np.pi/4, np.pi/4)

        self.linear_velocity = v
        self.angular_velocity = omega

        # Save previous distance to goal
        prev_distance = np.linalg.norm(self.goal - self.pose[:2])

        # Apply motion
        self.pose[2] += omega
        self.pose[2] = np.arctan2(np.sin(self.pose[2]), np.cos(self.pose[2]))
        self.pose[0] += v * np.cos(self.pose[2])
        self.pose[1] += v * np.sin(self.pose[2])

        # Compute new distance
        new_distance = np.linalg.norm(self.goal - self.pose[:2])
        progress = prev_distance - new_distance  # positive if moving closer

        # Reward structure
        reward = 1.0 * progress                      # reward for getting closer
        reward -= 0.01 * abs(omega)                   # slight penalty for turning too much
        reward -= 0.001                               # small time penalty

        # Done if close to goal
        done = new_distance < 1.0
        if done:
            reward += 100.0  # bonus for reaching the goal

        # Keep inside map
        self.pose[0] = np.clip(self.pose[0], 0, self.grid_size[1] * self.resolution)
        self.pose[1] = np.clip(self.pose[1], 0, self.grid_size[0] * self.resolution)

        self.previous_action = v
        return self._get_obs(), reward, done, {}


    def render(self):
        surf = pygame.display.set_mode(self.window_size)
        surf.fill((0, 0, 50))

        # --- parameters for panels ---
        map_w = 600
        total_w, total_h = self.window_size
        panel_w = (total_w - map_w) // 2    # 200 if width is 1000
        map_h = total_h

        # --- 1) Draw the map on [0..map_w] ---
        cw = map_w / self.grid_size[1]
        ch = map_h / self.grid_size[0]
        for y, x in zip(*np.where(self.occ_grid)):
            pygame.draw.rect(surf, (100,100,100),
                            (x*cw, y*ch, cw, ch))
        # goal & robot on map
        gx = self.goal[0]/self.resolution * cw
        gy = self.goal[1]/self.resolution * ch
        pygame.draw.circle(surf, self.goal_color,
                        (int(gx),int(gy)),
                        int(self.goal_radius/self.resolution*cw),2)
        x_pix = self.pose[0]/self.resolution*cw
        y_pix = self.pose[1]/self.resolution*ch
        pygame.draw.circle(surf,(0,255,0),(int(x_pix),int(y_pix)),5)
        ex = x_pix + 20*math.cos(self.pose[2])
        ey = y_pix + 20*math.sin(self.pose[2])
        pygame.draw.line(surf,(0,255,0),(int(x_pix),int(y_pix)),(int(ex),int(ey)),2)

        # get your ranges/intensities/mask
        ranges, intensities, hit_mask = self._get_raw_obs()

        ping_idx = None
        if self.beacon_enabled and (self.time % self.ping_interval) < self.pulse_duration:
            dx, dy = self.goal - self.pose[:2]
            bearing = math.atan2(dy, dx) - self.pose[2]
            bearing = (bearing + math.pi) % (2*math.pi) - math.pi
            ping_idx = np.argmin(np.abs(self.sonar.beam_angles - bearing))

        for i, (r, rel_ang) in enumerate(zip(ranges, self.sonar.beam_angles)):
            ang = self.pose[2] + rel_ang
            x2 = x_pix + (r/self.resolution)*cw * math.cos(ang)
            y2 = y_pix + (r/self.resolution)*ch * math.sin(ang)
            pygame.draw.line(surf, (0,200,200), (x_pix, y_pix), (x2, y2), 1)



        # --- 2) Fan‐beam Sonar Panel on [map_w..map_w+panel_w] ---
        sx0 = map_w
        sw = panel_w
        pygame.draw.rect(surf, (20,20,80), (sx0, 0, sw, total_h))
        bs = sw / self.sonar.n_beams
        # true hits
        for i, (r, inten, hit) in enumerate(zip(ranges,
                                            intensities if intensities is not None else [None]*len(ranges),
                                            hit_mask)):
            if not hit: continue
            px = sx0 + i*bs + bs/2
            py = total_h - (r/self.sonar.max_range)*(total_h-20)
            if inten is not None:
                norm = min(max(inten,0),1)
                col = (int(norm*255), 0, int((1-norm)*255))
            else:
                col = (0,200,200)
            pygame.draw.circle(surf, col, (int(px),int(py)), 6)
            
        # debris & ghost echoes
        for i, (r, inten, hit) in enumerate(zip(ranges, intensities if intensities is not None else [None]*len(ranges), hit_mask)):
            if not hit:
                continue
            # existing dot
            px = sx0 + i*bs + bs/2
            py = total_h - (r/self.sonar.max_range)*(total_h - 20)
            if inten is not None:
                norm = min(max(inten, 0.0), 1.0)
                color = (int(norm*255), 0, int((1-norm)*255))
            else:
                color = (0,200,200)
            pygame.draw.circle(surf, color, (int(px), int(py)), 6)

            # new: bold ping highlight
            if ping_idx == i:
                pygame.draw.circle(surf, (255,0,0), (int(px), int(py)), 10, 3)


        # 3) Cartesian Sonar Display to the right
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
            # flip X so ahead is upward, left is left
            px = center_x + dx * scale
            py = center_y - dy * scale

            if hit_mask[i]:
                col, rad = (255,255,0), 3
            else:
                col, rad = (50,50,50), 2
            pygame.draw.circle(surf, col, (int(px), int(py)), rad)

            # highlight docking ping
            if ping_idx == i:
                pygame.draw.circle(surf, (255,0,0), (int(px), int(py)), rad+3, 2)
        
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
