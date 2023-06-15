import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
  os.makedirs(models_dir)

if not os.path.exists(logdir):
  os.makedirs(logdir)

#TIMESTEPS = 100000
TIMESTEPS = 10000
env = make_vec_env('GoOne-v1', n_envs=500)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

for i in range(1, 250):
  model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
  model.save(f"{models_dir}/{TIMESTEPS*i}")
