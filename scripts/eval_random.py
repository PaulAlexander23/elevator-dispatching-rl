"""Evaluate hand-written baseline policies (random and up/down sweep)."""

from gymnasium.spaces import Discrete
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from elevator_rl.env import LiftEnv


class RandomPolicy:
    def predict(self, observations, state, episode_start, deterministic):
        action_space = Discrete(3)
        action = [action_space.sample() for n in range(len(observations))]
        return action, state


class UpDownPolicy:
    """Sweep the building, serving every floor. Expects obs_type="box"."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.going_up = True
        self.served = False

    def predict(self, observations, state, episode_start, deterministic):
        if episode_start is not None and episode_start[0]:
            self.reset()

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


def main():
    eval_envs = DummyVecEnv([lambda: Monitor(LiftEnv("human"))])
    for policy in (RandomPolicy(), UpDownPolicy()):
        mean_reward, std_reward = evaluate_policy(
            policy, eval_envs, n_eval_episodes=1000)
        print(f"{type(policy).__name__}: "
              f"mean reward: {mean_reward}, std reward: {std_reward}")


if __name__ == "__main__":
    main()
