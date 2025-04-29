# scripts/train_random.py

import numpy as np
from environments.simple_env import simpleAUVEnv

def main():
    env = simpleAUVEnv(
        sonar_params   = {'compute_intensity': False},
        current_params = None,
        goal_params    = {'radius': 0.5},
        beacon_params  = {
            'ping_interval':   1.0,
            'pulse_duration':  0.1,
            'beacon_intensity': 1.0,
            'ping_noise':      0.01
        }
    )

    obs = env.reset()
    total_steps = 0
    total_episodes = 0
    total_reward = 0.0

    while total_steps < 5000:
        action = np.random.uniform(low=[-1.0, -1.0], high=[1.0, 1.0])
        obs, reward, done, _ = env.step(action)

        assert np.isfinite(reward), f"[ERROR] Non-finite reward at step {total_steps}: {reward}"
        total_reward += reward
        total_steps += 1

        if done:
            print(f"[Episode {total_episodes}] Total Reward: {total_reward:.3f}, Steps: {total_steps}")
            obs = env.reset()
            total_reward = 0.0
            total_episodes += 1

if __name__ == "__main__":
    main()
