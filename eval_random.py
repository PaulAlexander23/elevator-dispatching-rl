from env import LiftEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.spaces import Discrete


class MyPolicy:
    def predict(self, observations, state, episode_start, deterministic):
        action_space = Discrete(3)
        action = [action_space.sample() for n in range(len(observations))]
        return action, state


class UpDownPolicy:
    def __init__(self):
        self.going_up = True
        self.served = False

    def predict(self, observations, state, episode_start, deterministic):

        if observations[0][0] == 1 and self.going_up:
            self.going_up = False

        if observations[0][0] == 0 and not self.going_up:
            self.going_up = True

        if not self.served:
            action = 2
            self.served = True
        else:
            self.served = False
            if self.going_up:
                action = 0
            else:
                action = 1
        return [action], state


eval_envs = DummyVecEnv([lambda: Monitor(LiftEnv("human"))])
# evaluate_policy(UpDownPolicy(), eval_envs, render=True, n_eval_episodes=1)
mean_reward, std_reward = evaluate_policy(
    UpDownPolicy(), eval_envs, n_eval_episodes=1000)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")
