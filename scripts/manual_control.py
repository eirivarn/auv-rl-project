# scripts/manual_control.py

import sys
import pygame
from environments.simple_env import simpleAUVEnv
import numpy as np

def main():
    # 1) Init Pygame
    pygame.init()

    # 2) Create env with ONLY the parameters your new __init__ expects:
    env = simpleAUVEnv(
        grid_size      = (200, 200),
        resolution     = 0.05,
        sonar_params   = {
            'fov'         : np.deg2rad(90),
            'n_beams'     : 60,
            'max_range'   : 10.0,
            'resolution'  : 0.05,
            'noise_std'   : 0.0,
            'compute_intensity': False,
            'debris_rate' : 0,
            'ghost_prob'  : 0.0
        },
        docks          = 1,             # one random dock
        dock_radius    = 0.5,
        dock_reward    = 1000,
        use_history    = False,         # or True if you want to test history
        history_length = 3,
        window_size    = (1000, 600)
    )

    obs = env.reset()
    clock = pygame.time.Clock()
    running = True

    while running:
        # 3) Handle quit
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # 4) Keyboard control
        keys = pygame.key.get_pressed()
        v     =  0.1 if keys[pygame.K_UP]   else -0.1 if keys[pygame.K_DOWN]  else 0.0
        omega =  0.1 if keys[pygame.K_LEFT] else -0.1 if keys[pygame.K_RIGHT] else 0.0

        # 5) Step the env
        obs, _, done, _ = env.step((v, omega))
        if done:
            obs = env.reset()

        # 6) Render and tick
        env.render()
        clock.tick(30)

    # 7) Clean up
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
