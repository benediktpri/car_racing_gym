import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from utils import load_model, run_single_episode
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your environment and preprocessing
env = gym.make("CarRacing-v2", render_mode='human', continuous=False)
env = GrayScaleObservation(env)
env = ResizeObservation(env, (84, 84))
env = FrameStack(env, 4)

# Load the trained model
model_path = 'reinforcement_learning/models/kaggle/20240708_reinforcement_learning/models/training_20240708_181819/final_dqn_model.pth'

num_actions = env.action_space.n
model = load_model(model_path, num_actions)

# Test the trained agent
run_single_episode(env, model)
