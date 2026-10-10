"""Benchmark batched env throughput: the C++ VecEnv against EnvPool.

Random actions are drawn up front, so only stepping is timed (no policy).
Rows:
- cpp: CppVecEnv, every env stepped in one thread;
- envpool sync: all envs stepped on `threads` threads, one batch per step;
- envpool async: the same pool with batch_size = n_envs / 2, so Python gets
  whichever half finishes first and sends their actions while the rest run.

    uv run python scripts/time_vec_env.py --n-envs 16 64 256 1024 --threads 1 2 4 8
"""

import argparse
import time

import numpy as np

from elevator_rl.cpp_env import CppVecEnv
from elevator_rl.envpool_env import EnvPoolVecEnv, envpool, envpool_kwargs


def actions_for(space, n_envs, rounds, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(space.nvec[0], size=(rounds, n_envs, len(space.nvec)), dtype=np.int32)


def time_vec_env(venv, steps):
    actions = actions_for(venv.action_space, venv.num_envs, 64)
    venv.reset()
    t0 = time.perf_counter()
    for i in range(steps):
        venv.step(actions[i % 64])
    return steps * venv.num_envs / (time.perf_counter() - t0)


def time_async(preset, obs_type, n_envs, threads, steps):
    """Raw async loop: recv a half batch, send its actions back."""
    reference = EnvPoolVecEnv(preset, 1, obs_type)._reference
    batch = n_envs // 2
    pool = envpool.make(
        "Elevator-v0",
        env_type="gymnasium",
        num_envs=n_envs,
        batch_size=batch,
        num_threads=threads,
        **envpool_kwargs(reference),
    )
    actions = actions_for(reference.action_space, batch, 64)
    pool.async_reset()
    t0 = time.perf_counter()
    for i in range(2 * steps):  # two half batches = one step of every env
        _, _, _, _, info = pool.recv()
        pool.send(actions[i % 64], info["env_id"])
    rate = steps * n_envs / (time.perf_counter() - t0)
    pool.close()
    return rate


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="full")
    parser.add_argument("--obs-type", default="relative", choices=("box", "relative"))
    parser.add_argument("--n-envs", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--env-steps", type=int, default=2_000_000, help="per row")
    args = parser.parse_args()

    print("| VecEnv | n_envs | Threads | env steps/s |")
    print("|---|---:|---:|---:|")
    for n_envs in args.n_envs:
        steps = max(args.env_steps // n_envs, 50)
        venv = CppVecEnv(args.preset, n_envs, args.obs_type)
        print(f"| cpp | {n_envs} | 1 | {time_vec_env(venv, steps):,.0f} |", flush=True)
        for threads in args.threads:
            venv = EnvPoolVecEnv(args.preset, n_envs, args.obs_type, num_threads=threads)
            rate = time_vec_env(venv, steps)
            venv.close()
            print(f"| envpool sync | {n_envs} | {threads} | {rate:,.0f} |", flush=True)
        for threads in args.threads:
            rate = time_async(args.preset, args.obs_type, n_envs, threads, steps)
            print(f"| envpool async | {n_envs} | {threads} | {rate:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
