import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os
from my_ant import My_AntEnv

models_dir = "models/PPO"
model_path = f"{models_dir}/2320000.zip"

TIMESTEPS = 100000
env = make_vec_env('GoOne-v1')
model = PPO.load(model_path, env=env)

obs = env.reset()
while True:
    env.render()
    action, _ = model.predict(obs)
    obs, rewards, done, info = env.step(action)
env.close()



"""env = make_vec_env('Ant-v5')

print(env.action_space.sample())


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=3000000)

#Sets an initial state
obs = env.reset()
# Rendering our instance 300 times
while True:
  action, _ = model.predict(obs)
  obs, rewards, done, info = env.step(action)

  env.render()
env.close()"""
