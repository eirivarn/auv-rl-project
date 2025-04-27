# scripts/manual_control.py

import sys
import pygame
from environments.auv_env import AUVEnv

def main():
    # Initialize Pygame before any display or event calls
    pygame.init()

    env = AUVEnv(
        sonar_params   = {'compute_intensity': False},
        current_params = None,    # disable currents
        goal_params    = {'radius': 0.5},
        beacon_params  = {
            'ping_interval':   1.0,   # seconds between beacon chirps
            'pulse_duration':  0.1,   # active listening window
            'beacon_intensity': 1.0,  # override intensity
            'ping_noise':      0.01   # noise on the beacon range
        }
    )

    obs = env.reset()
    clock = pygame.time.Clock()
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        v     =  1.0 if keys[pygame.K_UP]   else -1.0 if keys[pygame.K_DOWN]  else 0.0
        omega =  1.0 if keys[pygame.K_LEFT] else -1.0 if keys[pygame.K_RIGHT] else 0.0

        obs, _, done, _ = env.step((v, omega))
        if done:
            obs = env.reset()

        env.render()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()