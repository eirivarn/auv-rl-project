# scripts/manual_control.py

import sys
import pygame
import numpy as np
from environments.realistic_env import realisticAUVEnv  # <-- use your realistic env

def main():
    pygame.init()

    env = realisticAUVEnv(
        # --- simpleAUVEnv kwargs ---
        grid_size      = (200, 200),
        resolution     = 0.05,
        docks          = 1,
        dock_radius    = 0.5,
        dock_reward    = 1000,
        use_history    = False,
        history_length = 3,
        window_size    = (1000, 600),

        # --- physics kwargs ---
        mass           = 1.0,
        drag_coef      = 0.1,
        current_params = {
            'strength': 0.2,
            'period'  : 30.0,
            'direction': np.deg2rad(45)
        },
        dt             = 0.1,

        # --- this line lets you pass (v,omega) directly ---
        discrete_actions = False
    )

    obs = env.reset()
    clock = pygame.time.Clock()
    running = True

    while running:
        # 1) Handle quit
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # 2) Read keys
        keys = pygame.key.get_pressed()
        v     =  0.1 if keys[pygame.K_UP]   else -0.1 if keys[pygame.K_DOWN]  else 0.0
        omega =  0.1 if keys[pygame.K_LEFT] else -0.1 if keys[pygame.K_RIGHT] else 0.0

        # 3) Step with continuous commands
        obs, _, done, _ = env.step((v, omega))
        if done:
            obs = env.reset()

        # 4) Render and cap framerate
        env.render()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
