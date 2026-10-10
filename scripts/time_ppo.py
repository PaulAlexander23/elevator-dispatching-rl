"""Benchmark PPO training throughput. Needs the `bench` extra."""

import argparse
import os
import time

import psutil
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from elevator_rl.building import PRESETS
from elevator_rl.train import make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=PRESETS, help="time a BuildingEnv preset instead")
    args = parser.parse_args()
    # BuildingEnv has no multi_discrete obs; custom is its closest equivalent.
    obs_type = "custom" if args.preset else "multi_discrete"
    process = psutil.Process(os.getpid())
    envs = DummyVecEnv([lambda: make_env(obs_type, preset=args.preset)])

    model = PPO("MlpPolicy", envs, device="cpu", verbose=1)
    for chunk in range(20):
        t0 = time.perf_counter()

        model.learn(total_timesteps=100_000, log_interval=10, reset_num_timesteps=False)

        dt = time.perf_counter() - t0

        print(
            f"steps={(chunk + 1) * 100_000:7d} "
            f"FPS={100_000 / dt:8.1f} "
            f"RAM: {process.memory_info().rss / 1024**3:.2f} GB "
        )


if __name__ == "__main__":
    main()
