import matplotlib.pyplot as plt
from typing import List
from environments.grid_env import GridDockEnv
from agents.q_learning_agent import QLearningAgent
from IPython.display import clear_output, display
import numpy as np
from matplotlib import animation
import matplotlib.patches as patches
from IPython.display import HTML

def train_dqn(env, agent, episodes=1000, max_steps=100):
    rewards_hist = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state)
            next_s, reward, done, _ = env.step(action)

            # you can apply reward shaping here if desired
            agent.store_transition(state, action, reward, next_s, done)
            agent.optimize_model()

            state = next_s
            total_reward += reward
            agent.step_counter += 1
            agent.maybe_update_target()

            if done:
                break

        agent.update_epsilon()
        rewards_hist.append(total_reward)

    return rewards_hist


def train_agent(env: GridDockEnv,
                agent: QLearningAgent,
                episodes: int = 500,
                max_steps: int = 100,
                save_path: str = None,
                decay_epsilon: bool = True) -> (List[float], List[float]):
    """
    Train a Q-learning agent and log rewards and epsilon history.
    Returns:
        rewards_history: list of total rewards per episode
        eps_history: list of epsilon values per episode
    """
    rewards_history = []
    eps_history = []

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            if done:
                break
        if decay_epsilon:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        rewards_history.append(total_reward)
        eps_history.append(agent.epsilon)
    if save_path is not None:
        agent.save(save_path)
    return rewards_history, eps_history

def plot_rewards(rewards: List[float]):
    """Plot total reward per episode."""
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning: Episode Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_epsilon(eps_history: List[float]):
    """Plot epsilon decay over episodes."""
    plt.figure()
    plt.plot(eps_history)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def animate_agent_matplotlib(env, agent, max_steps: int = 100, delay: float = 0.1, figsize: tuple = (5,5)):
    """
    Simulate one greedy episode and return an inline HTML5 animation
    of the agent moving on the grid (no pygame).

    Args:
        env: an environment with .reset(), .step(), and attributes:
             env.grid_size (tuple), env.agent_pos, env.goal_pos.
        agent: any agent with .select_action(obs) and .epsilon attribute.
        max_steps: maximum steps to simulate in case it never reaches the goal.
        delay: seconds between frames.
        figsize: (width, height) in inches for the figure.

    Returns:
        IPython.display.HTML — the HTML5 animation.
    """
    # 1) simulate trajectory
    agent.epsilon = 0.0
    obs = env.reset()
    agent_positions = [env.agent_pos.copy()]
    goal_pos = env.goal_pos.copy()
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.select_action(obs)
        obs, _, done, _ = env.step(action)
        agent_positions.append(env.agent_pos.copy())
        steps += 1

    # 2) set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    W, H = env.grid_size
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.grid(True)

    # draw goal once
    gx, gy = goal_pos
    goal_patch = patches.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color='green')
    ax.add_patch(goal_patch)

    # draw agent
    agent_patch = patches.Circle((agent_positions[0][0], agent_positions[0][1]), 0.3, color='blue')
    ax.add_patch(agent_patch)

    # animation update func
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

def evaluate_agent(env, agent, episodes: int = 100, max_steps: int = 100):
    """
    Run the agent (greedy policy) for `episodes` random episodes,
    and return (success_rate, avg_steps_on_success).
    """
    # Force greedy
    agent.epsilon = 0.0

    successes = 0
    steps_list = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(obs)
            obs, _, done, _ = env.step(action)
            steps += 1

        if done:
            successes += 1
            steps_list.append(steps)

    success_rate = successes / episodes
    avg_steps    = sum(steps_list) / len(steps_list) if steps_list else None
    return success_rate, avg_steps


