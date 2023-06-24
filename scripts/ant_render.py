import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os
from my_ant import My_AntEnv

#models_dir = "models/PPO"
models_dir = "logs/last_model"
model_path = f"{models_dir}/334848000.zip"

TIMESTEPS = 100000
env = make_vec_env('Ant-v5')
model = PPO.load(model_path, env=env)

obs = env.reset()
while True:
    env.render()
    action, _ = model.predict(obs)
    obs, rewards, done, info = env.step(action)
env.close()
