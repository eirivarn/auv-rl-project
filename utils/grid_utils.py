import matplotlib.pyplot as plt
from typing import List
from environments.grid_env import GridEnv
from agents.q_learning_agent import QLearningAgent
from IPython.display import clear_output, display
import numpy as np
from matplotlib import animation
import matplotlib.patches as patches
from IPython.display import HTML
from tqdm import tqdm


def train_dqn(env, agent, episodes=1000, max_steps=100):
    """
    Train a DQN agent with a tqdm progress bar showing episode reward and epsilon.
    """
    rewards_hist = []

    # Create a tqdm iterator over the episode indices
    pbar = tqdm(range(episodes), desc="DQN Training")
    for ep in pbar:
        state, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):
            action = agent.select_action(state)
            next_s, reward, done, _ = env.step(action)

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

        # Update the bar’s postfix fields
        pbar.set_postfix({
            "Reward": f"{total_reward:.1f}",
            "ε":      f"{agent.epsilon:.3f}"
        })

    return rewards_hist

def train_agent(env: GridEnv,
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
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            obs, _ = reset_output
        else:
            obs = reset_output

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

def plot_rewards(rewards: List[float], window: int = 20):
    plt.figure()
    episodes = np.arange(len(rewards))

    # Plot raw rewards lightly
    plt.plot(episodes, rewards, color='C0', alpha=0.3, label='Raw Reward')

    # Compute moving average
    if window > 1 and len(rewards) >= window:
        # 'valid' gives us len(rewards)-window+1 points
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ma_eps = np.arange(window-1, len(rewards))
        plt.plot(ma_eps, ma, color='C1', label=f'{window}-Episode MA')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.legend()
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
        obs, _ = env.reset()
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


