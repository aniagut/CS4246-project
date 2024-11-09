import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sumo_env import SumoEnv
from stable_baselines3.common.callbacks import BaseCallback

# Custom callback to track total rewards per episode during training
class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_reward = 0

    def _on_step(self):
        reward = self.locals["rewards"][0]
        self.episode_reward += reward
        done = self.locals["dones"][0]
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        return True

# Function to run PPO with different hyperparameters
def run_experiment(batch_size, gamma, clip_range, n_steps):
    env = SumoEnv(config_file="intersection.sumocfg", max_steps=500)
    model = PPO(
        "MlpPolicy",
        env,
        batch_size=batch_size,
        gamma=gamma,
        clip_range=clip_range,
        n_steps=n_steps,
        verbose=1,
    )
    callback = RewardTrackingCallback()
    model.learn(total_timesteps=50000, callback=callback)
    env.close()
    return callback.episode_rewards

# Parameter configurations to experiment with
batch_sizes = [32, 64, 128]
gammas = [0.9, 0.99, 0.995]
clip_ranges = [0.1, 0.2]
n_steps_list = [128, 256, 512]

# Plot results for each parameter

# Experiment with different batch sizes
plt.figure(figsize=(10, 6))
for batch_size in batch_sizes:
    rewards = run_experiment(batch_size=batch_size, gamma=0.99, clip_range=0.2, n_steps=256)
    plt.plot(rewards, label=f"Batch Size={batch_size}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode for Different Batch Sizes")
plt.legend()
plt.show()

# Experiment with different gamma values
plt.figure(figsize=(10, 6))
for gamma in gammas:
    rewards = run_experiment(batch_size=64, gamma=gamma, clip_range=0.2, n_steps=256)
    plt.plot(rewards, label=f"Gamma={gamma}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode for Different Gamma Values")
plt.legend()
plt.show()

# Experiment with different clip ranges
plt.figure(figsize=(10, 6))
for clip_range in clip_ranges:
    rewards = run_experiment(batch_size=64, gamma=0.99, clip_range=clip_range, n_steps=256)
    plt.plot(rewards, label=f"Clip Range={clip_range}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode for Different Clip Ranges")
plt.legend()
plt.show()

# Experiment with different n_steps values
plt.figure(figsize=(10, 6))
for n_steps in n_steps_list:
    rewards = run_experiment(batch_size=64, gamma=0.99, clip_range=0.2, n_steps=n_steps)
    plt.plot(rewards, label=f"n_steps={n_steps}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode for Different n_steps Values")
plt.legend()
plt.show()