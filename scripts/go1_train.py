import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

import time

import os

n_training_envs = 350
n_eval_envs = 5
TIMESTEPS = 500000000

class CheckPointCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(CheckPointCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path_best = os.path.join(log_dir, "best_model")
        self.save_path_last = os.path.join(log_dir, "last_model")
        self.best_mean_reward = -np.inf
        self.init_time = time.time()
        self.epoch_time = self.init_time
        self.freq = check_freq
        self.checks_left = check_freq

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path_best is not None:
            os.makedirs(self.save_path_best, exist_ok=True)
        if self.save_path_last is not None:
            os.makedirs(self.save_path_last, exist_ok=True)

        print("Callback Running...")

    def _on_step(self) -> bool:
        self.checks_left = self.checks_left - 1
        #if self.n_calls % self.check_freq == 0:

        Stop = False
        if self.checks_left == 0:
          self.checks_left = self.freq
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                actual_time = time.time()
                d1 = (actual_time - self.epoch_time) / 60
                d2 = (actual_time - self.init_time) / 60

                d3 = ( d1 - int(d1) ) * 60
                d4 = ( d2 - int(d2) ) * 60
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Time since last update: {int(d1)} min {d3} sec")
                print(f"Total time: {int(d2)} min {d4} sec")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
                print(f"Saving latest model to {self.save_path_last}")

                self.epoch_time = actual_time

              self.model.save(f"{self.save_path_last}/{self.num_timesteps}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path_best}")
                  self.model.save(f"{self.save_path_best}/best_model")
              elif self.verbose >= 1:
                 print("Recent model does not improve the best model")

              f = open("EarlyStopping", "r")
              data = f.read()

              print("EarlyStopping: ", data)

              Stop = True if data == "1" else False
              
              if data == "1":
                 f = open("EarlyStopping", "w")
                 f.write("0")
                 f.close()

        return not Stop

models_dir = "models/PPO"
logdir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir,     exist_ok=True)

train_env = make_vec_env('GoOne-v1', n_envs=n_training_envs, seed=0)
train_env = VecMonitor(train_env, logdir)

model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=logdir)

callback = CheckPointCallback(check_freq=2048, log_dir=logdir)

model.learn(total_timesteps=int(TIMESTEPS), tb_log_name="PPO", callback=callback)
