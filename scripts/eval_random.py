"""Evaluate hand-written baseline policies (random and up/down sweep)."""

from stable_baselines3.common.evaluation import evaluate_policy

from elevator_rl.baselines import RandomPolicy, UpDownPolicy
from elevator_rl.train import make_eval_env


def main():
    eval_envs = make_eval_env("box", render_mode="human")
    for policy in (RandomPolicy(), UpDownPolicy()):
        mean_reward, std_reward = evaluate_policy(policy, eval_envs, n_eval_episodes=1000)
        print(f"{type(policy).__name__}: mean reward: {mean_reward}, std reward: {std_reward}")


if __name__ == "__main__":
    main()
