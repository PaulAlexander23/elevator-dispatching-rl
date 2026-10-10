"""Benchmark the JAX env's throughput: random actions, all inside one jit.

Each row runs `--steps` steps of `n_envs` envs as one `lax.scan`, so there
is no Python in the loop at all. The first call compiles and is not timed.
Needs the `jax` dependency group (`jax-cuda` for a GPU).

    uv run --group jax-cuda python scripts/time_jax_env.py --n-envs 1024 4096 16384
"""

import argparse
import time

import jax

from elevator_rl.jax_env import JaxBuildingEnv, make_vec_env


def throughput(env, n_envs, steps, device):
    reset, step = make_vec_env(env, n_envs)

    def run(key):
        vec_state, _ = reset(key)

        def body(vec_state, key):
            actions = jax.random.randint(key, (n_envs, env.config.n_lifts), 0, env.n_actions)
            vec_state, obs, reward, done, _ = step(vec_state, actions)
            return vec_state, reward.sum()

        vec_state, rewards = jax.lax.scan(body, vec_state, jax.random.split(key, steps))
        return rewards.sum()

    run = jax.jit(run, device=device)
    run(jax.random.key(0)).block_until_ready()  # compile
    t0 = time.perf_counter()
    run(jax.random.key(1)).block_until_ready()
    return n_envs * steps / (time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="full")
    parser.add_argument("--obs-type", default="relative", choices=("box", "relative"))
    parser.add_argument("--n-envs", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    parser.add_argument("--steps", type=int, default=400, help="steps per env (2 episodes)")
    parser.add_argument("--devices", nargs="+", default=["gpu", "cpu"])
    args = parser.parse_args()

    env = JaxBuildingEnv(args.preset, reward_shaping=True, obs_type=args.obs_type)
    print("| Device | n_envs | env steps/s |")
    print("|---|---:|---:|")
    for name in args.devices:
        try:
            device = jax.devices(name)[0]
        except RuntimeError:
            print(f"| {name} | - | not available |")
            continue
        for n_envs in args.n_envs:
            rate = throughput(env, n_envs, args.steps, device)
            print(f"| {name} | {n_envs} | {rate:,.0f} |", flush=True)


if __name__ == "__main__":
    main()
