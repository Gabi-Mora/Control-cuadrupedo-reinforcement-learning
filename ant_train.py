import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import os

models_dir = "models/PPO"
logdir = "logs"

#env = gym.make('Humanoid-v3')
#env = make_vec_env('Ant-v3')
#env = My_AntEnv()

if not os.path.exists(models_dir):
  os.makedirs(models_dir)

if not os.path.exists(logdir):
  os.makedirs(logdir)

#from my_ant import My_AntEnv

#TIMESTEPS = 100000
TIMESTEPS = 10000
env = make_vec_env('GoOne-v1', n_envs=500)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

for i in range(1, 250):
  model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
  model.save(f"{models_dir}/{TIMESTEPS*i}")


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
