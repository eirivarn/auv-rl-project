import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from IPython.display import HTML
import numpy as np
import pygame
from tqdm import trange

def animate_agent_matplotlib(env, agent, max_steps: int = 100, delay: float = 0.1, figsize: tuple = (5,5)):
    agent.epsilon = 0.0
    obs, _ = env.reset()
    agent_positions = [tuple(env.agent_position)]
    goal_position = tuple(env.goal_position)
    obstacles = set(getattr(env, "obstacles", []))
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.select_action(obs)
        obs, _, done, _ = env.step(action)
        agent_positions.append(tuple(env.agent_position))
        steps += 1

    # 2) set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    W, H = env.grid_size
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.grid(True)

    for (ox, oy) in obstacles:
        rect = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, color='black')
        ax.add_patch(rect)

    gx, gy = goal_position
    goal_patch = patches.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color='green')
    ax.add_patch(goal_patch)

    agent_patch = patches.Circle((agent_positions[0][0], agent_positions[0][1]), 0.3, color='blue')
    ax.add_patch(agent_patch)

    def _update(frame_idx):
        x, y = agent_positions[frame_idx]
        agent_patch.center = (x, y)
        return (agent_patch,)

    ani = animation.FuncAnimation(
        fig, _update,
        frames=len(agent_positions),
        interval=delay * 1000,
        blit=True
    )
    plt.close(fig)
    return HTML(ani.to_jshtml())

def record_pygame_robust(env, agent, out_path='auv.avi', max_steps=200, fps=30):
    """
    Robustly record a pygame‐based run of `env` under `agent` to a video file.
    Handles end-of-episode cleanly by breaking before rendering/capture.
    """
    # 1) Initialize Pygame once
    pygame.init()
    width, height = env.window_size
    _ = pygame.display.set_mode((width, height))

    # 2) Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        pygame.quit()
        raise RuntimeError(f"Cannot open video writer for {out_path}")

    try:
        agent.epsilon = 0.0
        state, _ = env.reset()
        done = False

        for t in trange(max_steps, desc="Recording"):
            # 3) Step agent first, so we don't render after done
            idx = agent.select_action(state)
            state, _, done, _ = env.step(idx)
            if done:
                break

            # 4) Handle window events
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    done = True
                    break

            # 5) Render and capture
            env.render()
            surf = pygame.display.get_surface()
            if surf is None:
                break
            arr = pygame.surfarray.array3d(surf)  # (w,h,3)
            frame = cv2.cvtColor(np.transpose(arr, (1,0,2)), cv2.COLOR_RGB2BGR)
            writer.write(frame)

            # 6) Wait to target FPS
            pygame.time.wait(int(1000/fps))

    finally:
        writer.release()
        pygame.quit()

    print(f"Recording saved to {out_path}")

def record_headless(env, agent, out_path='auv.gif', max_steps=200, fps=10):
    import imageio
    frames = []

    agent.epsilon = 0.0
    state, _ = env.reset()

    done = False
    t = 0
    # capture initial frame
    frames.append(env.render(mode='rgb_array'))

    while not done and t < max_steps:
        # 1) choose action from current state
        idx   = agent.select_action(state)

        # 2) step the env
        obs, _, done, _ = env.step(idx)
        state           = obs   # ← **important**

        # 3) render after step so you see the result
        frames.append(env.render(mode='rgb_array'))

        t += 1

    # 4) write GIF
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Headless recording saved to {out_path}")

