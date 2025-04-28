import sys
import math
import random
import pygame
import numpy as np

# --------------------------------------
# Very Simple Particle Filter Demo
# --------------------------------------
# 600x600 window with 4 landmarks. Manual control of robot.
# Particles estimate pose via noisy range measurements.

WINDOW_SIZE = 600
NUM_PARTICLES = 300
LMARKS = [(100,100), (500,100), (100,500), (500,500)]  # pixel coords
VELOCITY = 2.0  # pixels per frame
OMEGA = 0.1     # radians per frame
SIGMA_MOVE = 1.0
SIGMA_TURN = 0.05
SIGMA_RANGE = 10.0  # pixels noise in measurement

class ParticleFilter:
    def __init__(self, n):
        self.n = n
        # particles: x, y, theta
        self.p = np.zeros((n,3))
        self.p[:,0] = np.random.uniform(0, WINDOW_SIZE, n)
        self.p[:,1] = np.random.uniform(0, WINDOW_SIZE, n)
        self.p[:,2] = np.random.uniform(0, 2*math.pi, n)
        self.w = np.ones(n)/n

    def predict(self, v, omega):
        # add noise to each particle
        v_noisy = np.random.normal(v, SIGMA_MOVE, self.n)
        o_noisy = np.random.normal(omega, SIGMA_TURN, self.n)
        th = self.p[:,2] + o_noisy
        x = self.p[:,0] + v_noisy * np.cos(th)
        y = self.p[:,1] + v_noisy * np.sin(th)
        # keep inside window
        x = np.clip(x, 0, WINDOW_SIZE)
        y = np.clip(y, 0, WINDOW_SIZE)
        th %= 2*math.pi
        self.p = np.column_stack((x,y,th))

    def update(self, measurements):
        # measurements: list of distances to landmarks
        weights = np.ones(self.n)
        for i, (lx,ly) in enumerate(LMARKS):
            dx = self.p[:,0] - lx
            dy = self.p[:,1] - ly
            dists = np.hypot(dx, dy)
            # compute Gaussian likelihood
            weights *= np.exp(-0.5 * ((dists - measurements[i]) / SIGMA_RANGE)**2)
        weights += 1e-300
        self.w = weights / np.sum(weights)

    def resample(self):
        # systematic resample
        positions = (np.arange(self.n) + random.random())/self.n
        indexes = np.zeros(self.n, 'i')
        cdf = np.cumsum(self.w)
        i, j = 0, 0
        while i < self.n:
            if positions[i] < cdf[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.p[:] = self.p[indexes]
        self.w.fill(1.0/self.n)

    def estimate(self):
        x = np.average(self.p[:,0], weights=self.w)
        y = np.average(self.p[:,1], weights=self.w)
        sin_t = np.average(np.sin(self.p[:,2]), weights=self.w)
        cos_t = np.average(np.cos(self.p[:,2]), weights=self.w)
        theta = math.atan2(sin_t, cos_t)
        return x, y, theta

class Robot:
    def __init__(self):
        self.x = WINDOW_SIZE/2
        self.y = WINDOW_SIZE/2
        self.theta = 0.0

    def move(self, v, omega):
        self.theta += omega
        self.x += v * math.cos(self.theta)
        self.y += v * math.sin(self.theta)
        self.x = max(0, min(WINDOW_SIZE, self.x))
        self.y = max(0, min(WINDOW_SIZE, self.y))

    def sense(self):
        # true ranges with noise
        ranges = []
        for lx,ly in LMARKS:
            d = math.hypot(self.x-lx, self.y-ly)
            ranges.append(d + np.random.normal(0, SIGMA_RANGE))
        return ranges

def draw_robot(screen, x, y, theta, color):
    pygame.draw.circle(screen, color, (int(x), int(y)), 6)
    # heading line
    ex = x + 15*math.cos(theta)
    ey = y + 15*math.sin(theta)
    pygame.draw.line(screen, color, (int(x),int(y)), (int(ex),int(ey)), 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()

    robot = Robot()
    pf = ParticleFilter(NUM_PARTICLES)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        v = VELOCITY if keys[pygame.K_UP] else -VELOCITY if keys[pygame.K_DOWN] else 0.0
        o = -OMEGA if keys[pygame.K_LEFT] else OMEGA if keys[pygame.K_RIGHT] else 0.0

        # move robot
        robot.move(v, o)
        # predict
        pf.predict(v, o)
        # measurement
        meas = robot.sense()
        pf.update(meas)
        pf.resample()
        est_x, est_y, est_th = pf.estimate()

        # draw
        screen.fill((20,20,20))
        # landmarks
        for lx,ly in LMARKS:
            pygame.draw.circle(screen, (0,0,255), (lx,ly), 8)
        # particles
        for px,py,pt in pf.p:
            pygame.draw.circle(screen, (180,180,180), (int(px),int(py)), 2)
        # robot true
        draw_robot(screen, robot.x, robot.y, robot.theta, (0,255,0))
        # robot estimate
        draw_robot(screen, est_x, est_y, est_th, (255,255,0))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
