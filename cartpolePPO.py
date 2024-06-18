import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os

log_dir = "gymgraphs/"
os.makedirs(log_dir, exist_ok=True)
env = gym.make("CartPole-v1")
env = Monitor(env, log_dir)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_cartpole_graph/")

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

model.learn(total_timesteps=10000, log_interval=4, callback=eval_callback)
model.save("ppo_cartpole2")
model = PPO.load("ppo_cartpole2", env=env)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

#tensorboard --logdir=./ppo_cartpole_graph/
