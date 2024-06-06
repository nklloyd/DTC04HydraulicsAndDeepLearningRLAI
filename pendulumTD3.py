import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3

env = gym.make("Pendulum-v1", render_mode="human")

model = TD3("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("td3_pendulum")

model = TD3.load("td3_pendulum")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
