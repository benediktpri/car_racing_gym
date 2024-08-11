import gymnasium as gym
env = gym.make("CarRacing-v2", render_mode=None)
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(10):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print("running")
    if terminated or truncated:
        observation, info = env.reset()

env.close()