import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
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
def run_dqn_experiment(learning_rate=5e-4, buffer_size=50000, batch_size=64, gamma=0.95, target_update_interval=500, train_freq=8):

    env = SumoEnv(config_file="intersection.sumocfg", max_steps=500)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=batch_size,
        gamma=gamma,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        verbose=1,
    )
    callback = RewardTrackingCallback()
    model.learn(total_timesteps=50000, callback=callback)
    model.save("DQN_MODEL3")
    env.close()
    return callback.episode_rewards


plt.figure(figsize=(10, 6))


reward = run_dqn_experiment()
plt.plot(reward)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.show()



