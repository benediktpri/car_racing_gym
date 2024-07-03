import torch
from dqn import DQN
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path, num_actions):
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def run_single_episode(env, model):
    state, _ = env.reset()
    state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device) / 255.0
    total_reward = 0
    done = False
    
    while not done:
        with torch.no_grad():
            action = model(state).max(1)[1].item()
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0).to(device) / 255.0
        state = next_state
        total_reward += reward

    print(f'Total reward: {total_reward}')

import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(episode_rewards, window=10, save_path=None):
    """
    Plots the rewards over episodes and the moving average of rewards.
    
    :param episode_rewards: List of rewards per episode
    :param window: Window size for moving average
    :param save_path: Path to save the plot. If None, the plot is shown instead of saved.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(episode_rewards)), moving_avg, label='Moving Average', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_epsilon(epsilon_values, save_path=None):
    """
    Plots the epsilon decay over episodes.
    
    :param epsilon_values: List of epsilon values per episode
    :param save_path: Path to save the plot. If None, the plot is shown instead of saved.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Over Time')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def update_plot(episode_rewards, window=10):
    """
    Updates the plot with new rewards data.

    :param episode_rewards: List of rewards per episode
    :param window: Window size for moving average
    """
    plt.clf()
    plt.plot(episode_rewards, label='Episode Reward')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(episode_rewards)), moving_avg, label='Moving Average', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.pause(0.01)  # Pause to update the plot

