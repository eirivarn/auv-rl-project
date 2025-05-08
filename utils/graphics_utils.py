import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from IPython.display import HTML

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
