# scripts/manual_control.py

import sys
import pygame
import numpy as np
from environments.realistic_env import realisticAUVEnv


def main():
    pygame.init()

    # Instantiate realistic env in continuous mode for manual control
    env = realisticAUVEnv(
        grid_size=(200, 200),
        resolution=0.05,
        docks=1,
        dock_radius=0.5,
        dock_reward=1000,
        use_history=False,
        history_length=3,
        window_size=(1000, 600),
        mass=1.0,
        drag_coef=0.5,
        current_params={
            'strength': 0.0,
            'period': 30.0,
            'direction': 0.0
        },
        dt=0.1,
        discrete_actions=False  # continuous action space
    )

    obs = env.reset()
    clock = pygame.time.Clock()
    running = True

    print(
        "Controls:\n"
        "  W/S: forward/back\n"
        "  A/D: strafe left/right\n"
        "  ←/→: yaw left/right\n"
        "  R: reset\n"
        "  Esc or window close: quit"
    )

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (
               e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                obs = env.reset()

        # Read keyboard
        keys = pygame.key.get_pressed()
        # forward/backward with W/S
        fwd =  0.3 if keys[pygame.K_w] else -0.3 if keys[pygame.K_s] else 0.0
        # strafe left/right with A/D
        lat =  0.3 if keys[pygame.K_d] else -0.3 if keys[pygame.K_a] else 0.0
        # yaw left/right with arrow keys
        yaw =  0.3 if keys[pygame.K_LEFT] else -0.3 if keys[pygame.K_RIGHT] else 0.0

        # Step env with 3-DoF action tuple
        obs, reward, done, info = env.step((fwd, lat, yaw))
        if done:
            obs = env.reset()

        # Render and cap frame rate
        env.render()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
