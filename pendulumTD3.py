import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os

log_dir = "gymgraphs/"
os.makedirs(log_dir, exist_ok=True)
env = gym.make("Pendulum-v1", render_mode="human")
env = Monitor(env, log_dir)

model = TD3("MlpPolicy", env, verbose=1, tensorboard_log="./td3_pendulum_graph/")

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

model.learn(total_timesteps=10000, log_interval=4)
model.save("td3_pendulum2")
model = TD3.load("td3_pendulum2", env=env)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

#tensorboard --logdir=./td3_pendulum_graph/
