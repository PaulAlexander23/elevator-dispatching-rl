"""Evaluate a trained PPO model.

    uv run python scripts/eval.py model.zip --obs-type custom
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from elevator_rl.env import OBS_TYPES, LiftEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="path to a saved PPO .zip")
    parser.add_argument("--obs-type", choices=OBS_TYPES, default="custom",
                        help="observation type the model was trained with")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--render", action="store_true",
                        help="watch one stochastic episode first")
    args = parser.parse_args()

    model = PPO.load(args.model)
    eval_envs = DummyVecEnv(
        [lambda: Monitor(LiftEnv("human", obs_type=args.obs_type))])
    if args.render:
        evaluate_policy(model, eval_envs, render=True,
                        n_eval_episodes=1, deterministic=False)
    for deterministic in (True, False):
        mean_reward, std_reward = evaluate_policy(
            model, eval_envs, n_eval_episodes=args.episodes,
            deterministic=deterministic)
        print(f"deterministic={deterministic}: "
              f"mean reward: {mean_reward}, std reward: {std_reward}")


if __name__ == "__main__":
    main()
