"""Benchmark raw environment throughput. Needs the `bench` extra."""

import argparse
import os
import time

import psutil

from elevator_rl.building import PRESETS
from elevator_rl.train import make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=PRESETS, help="time a BuildingEnv preset instead")
    args = parser.parse_args()
    # BuildingEnv has no multi_discrete obs; custom is its closest equivalent.
    obs_type = "custom" if args.preset else "multi_discrete"
    process = psutil.Process(os.getpid())
    env = make_env(obs_type, reward_shaping=True, preset=args.preset)
    obs, info = env.reset(seed=0)

    for chunk in range(20):
        t0 = time.perf_counter()

        for _ in range(100_000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()
        dt = time.perf_counter() - t0

        print(
            f"steps={(chunk + 1) * 100_000:7d} "
            f"FPS={100_000 / dt:8.1f} "
            f"RAM: {process.memory_info().rss / 1024**3:.2f} GB "
        )


if __name__ == "__main__":
    main()
