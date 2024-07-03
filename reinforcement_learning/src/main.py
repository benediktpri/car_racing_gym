# import and stuff
import os
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
import random
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from dqn import DQN
from datetime import datetime
from utils import plot_rewards, update_plot, plot_epsilon
import matplotlib.pyplot as plt
import math
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# new directory for this training session within reinforcement_learning/models/
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
training_dir = os.path.join('reinforcement_learning', 'models', f'training_{timestamp}')
os.makedirs(training_dir, exist_ok=True)

# setup for logging
log_filename = os.path.join(training_dir, 'training_log.txt')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')

# env setup and preprocessing
env = gym.make("CarRacing-v2", render_mode=None, continuous=False)
env = GrayScaleObservation(env)
env = ResizeObservation(env, (84, 84))
env = FrameStack(env, 4)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # Use deque for efficient FIFO operation

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)  # Add experience to the buffer

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device) / 255.0
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device) / 255.0
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        
        # Reshape states and next_states to match the expected input shape for Conv2d: [batch_size, channels, height, width]
        states = states.view(batch_size, 4, 84, 84)
        next_states = next_states.view(batch_size, 4, 84, 84)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)  # Return the current size of the buffer

# Agent Class
class Agent:
    def __init__(self, num_actions, buffer_size, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update_frequency):
        self.current_step = 0
        self.num_actions = num_actions
        self.gamma = gamma  # Discount factor for future rewards
        # self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_threshold = epsilon_start
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.policy_net = DQN(num_actions).to(device)
        self.target_net = DQN(num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to eval mode
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.target_update_frequency = target_update_frequency

    def select_action(self, state):
        # Update epsilon with new decay rate 1000
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        math.exp(-1. * self.current_step / self.epsilon_decay)
        self.epsilon_threshold = eps_threshold
        self.current_step += 1
        # Epsilon-greedy policy
        if random.random() < eps_threshold:#self.epsilon:
            return random.randrange(self.num_actions)  # Explore
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()  # Exploit: select action with highest Q-value

    def store_experience(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Get current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Get next Q-values from target network
        next_q_values = self.target_net(next_states).max(1)[0]
        next_q_values[dones.bool()] = 0.0  # Zero out the Q-values where the episode has ended
        # Compute target Q-values
        target_q_values = rewards + (self.gamma * next_q_values)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon (inital version)
        # self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # print("Epsilon set to: "+ str(self.epsilon))

    def update_target_network(self):
        """Update the weights of the target network to match those of the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Constants for the agent
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
# EPSILON_DECAY = 0.995
EPSILON_DECAY = 100000  # Control the rate of decay
TARGET_UPDATE = 1000  # Update target network every 1000 steps

# Initialize the agent
agent = Agent(num_actions=env.action_space.n, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
              gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY, target_update_frequency=TARGET_UPDATE)

# Enable interactive mode 
plt.ion()
plt.figure(figsize=(12, 6))

# main training loop
num_episodes = 1000  # Total number of episodes to train
save_interval = 200  # How often to save the model
print_every = 20     # How often to print the average reward
episode_rewards = []  # List to store total rewards for each episode
epsilon_values = []   # List to store epsilon values for each episode

for episode in range(num_episodes):
    state, _ = env.reset()  # Reset the environment and obtain the initial state
    state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device) / 255.0  # Process state for DQN input
    total_reward = 0
    done = False

    # Initial 50-step delay
    for _ in range(50):
        next_state, reward, done, truncated, _ = env.step(0)  # Perform a no-op action
        if done or truncated:
            break  # Exit the delay loop if the episode ends prematurely
        state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0).to(device) / 255.0  # Process state

    while not done:
        action = agent.select_action(state)  # Select an action using the policy net
        next_state, reward, done, truncated, _ = env.step(action)  # Execute the action
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0).to(device) / 255.0  # Process next_state
        done = done or truncated

        agent.store_experience(state, action, reward, next_state, done)  # Store experience in the buffer
        agent.learn()  # Allow the agent to learn from the buffer

        state = next_state  # Move to the next state
        total_reward += reward

        if agent.current_step % agent.target_update_frequency == 0:
            agent.update_target_network()
    
    # Update epsilon after each episode
    # agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
    # print(f"Epsilon set to: {agent.epsilon}")

    episode_rewards.append(total_reward)  # Append total reward for this episode
    epsilon_values.append(agent.epsilon_threshold)

    if episode % print_every == 0:
        avg_reward = np.mean(episode_rewards[-print_every:])
        print(f'Episode {episode}: Total reward = {total_reward}, Average reward = {avg_reward:.4f}, Epsilon = {agent.epsilon_threshold:.4f}')
        logging.info(f'Episode {episode}: Total reward = {total_reward}, Average reward = {avg_reward:.4f}, Epsilon = {agent.epsilon_threshold:.4f}')


    if episode % save_interval == 0:
        model_filename = os.path.join(training_dir, f'dqn_model_{episode}_episodes.pth')
        torch.save(agent.policy_net.state_dict(), model_filename)
        print(f'Model saved: {model_filename}')

    # Update the plot after each episode
    update_plot(episode_rewards, window=50)

# Save the final model
final_model_filename = os.path.join(training_dir, f'final_dqn_model.pth')
torch.save(agent.policy_net.state_dict(), final_model_filename)
print(f'Final model saved: {final_model_filename}')
logging.info(f'Final model saved: {final_model_filename}')

# Disable interactive mode
plt.ioff()

# Plotting the rewards at the end of training
reward_plot_filename = os.path.join(training_dir, 'training_rewards.png')
plot_rewards(episode_rewards, window=50, save_path=reward_plot_filename)
print(f'Training rewards plot saved: {reward_plot_filename}')
logging.info(f'Training rewards plot saved: {reward_plot_filename}')

# Plotting the epsilon values at the end of training
epsilon_plot_filename = os.path.join(training_dir, 'epsilon_decay.png')
plot_epsilon(epsilon_values, save_path=epsilon_plot_filename)
print(f'Epsilon decay plot saved: {epsilon_plot_filename}')
logging.info(f'Epsilon decay plot saved: {epsilon_plot_filename}')

print('Training complete')
logging.info('Training complete')
