"""EnvPool's XLA interface: stepping the C++ env from inside `jax.jit`.

`pool.xla()` returns (handle, recv, send, step) functions that XLA calls as
custom calls, so a `lax.scan` can step EnvPool's thread pool without leaving
compiled code, as the JAX and Warp envs do. This times that against EnvPool's
Python API and the JAX and Warp envs, with random actions drawn in JAX.

    uv run --group warp python scripts/time_envpool_xla.py
    JAX_PLATFORMS=cpu uv run --group warp python scripts/time_envpool_xla.py --jax-backends

It times JAX's default device; run it with JAX_PLATFORMS=cpu for the CPU.
(Picking the CPU with jax.default_device while a GPU is present hung in
EnvPool's custom call.)

EnvPool auto-resets one step after an episode ends (that step's action is
ignored), unlike jax_env.make_vec_env, which resets in the same step: a
learner on it must mask that step out.
"""

import argparse
import time

import jax
import numpy as np

from elevator_rl.envpool_env import EnvPoolVecEnv

N_STEPS = 400


def time_xla(n_envs, preset):
    venv = EnvPoolVecEnv(preset, n_envs=n_envs)
    pool = venv._pool
    handle, _, _, step = pool.xla()
    # EnvPool's own action space is a Box; the VecEnv's is the MultiDiscrete one.
    n_lifts, n_actions = len(venv.action_space.nvec), int(venv.action_space.nvec[0])

    def body(carry, _):
        h, key = carry
        key, sub = jax.random.split(key)
        actions = jax.random.randint(sub, (n_envs, n_lifts), 0, n_actions)
        h, (obs, reward, *_rest) = step(h, actions)
        return (h, key), reward.sum()

    run = jax.jit(lambda h, k: jax.lax.scan(body, (h, k), None, length=N_STEPS))
    pool.async_reset()
    jax.block_until_ready(run(handle, jax.random.key(0)))  # compile
    t0 = time.perf_counter()
    jax.block_until_ready(run(handle, jax.random.key(1)))
    return n_envs * N_STEPS / (time.perf_counter() - t0)


def time_python(n_envs, preset):
    venv = EnvPoolVecEnv(preset, n_envs=n_envs)
    venv.reset()
    rng = np.random.default_rng(0)
    nvec = venv.action_space.nvec
    actions = [rng.integers(nvec, size=(n_envs, len(nvec))) for _ in range(N_STEPS)]
    t0 = time.perf_counter()
    for a in actions:
        venv.step(a)
    return n_envs * N_STEPS / (time.perf_counter() - t0)


def time_jax_env(backend, n_envs, preset):
    if backend == "warp":
        from elevator_rl.warp_env import WarpBuildingEnv as cls
    else:
        from elevator_rl.jax_env import JaxBuildingEnv as cls
    env = cls(preset)
    reset, step = env.make_vec_env(n_envs)
    n_lifts = env.config.n_lifts

    def body(carry, _):
        state, key = carry
        key, sub = jax.random.split(key)
        actions = jax.random.randint(sub, (n_envs, n_lifts), 0, env.n_actions)
        state, obs, reward, done, info = step(state, actions)
        return (state, key), reward.sum()

    run = jax.jit(lambda s, k: jax.lax.scan(body, (s, k), None, length=N_STEPS))
    state, _ = reset(jax.random.key(0))
    jax.block_until_ready(run(state, jax.random.key(0)))
    t0 = time.perf_counter()
    jax.block_until_ready(run(state, jax.random.key(1)))
    return n_envs * N_STEPS / (time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="full")
    parser.add_argument("--n-envs", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--jax-backends", nargs="*", default=["jax", "warp"])
    args = parser.parse_args()

    device = jax.devices()[0].platform
    head = ["n_envs", "EnvPool (Python API)", f"EnvPool in jit ({device})"]
    head += [f"{b} env ({device})" for b in args.jax_backends]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for n in args.n_envs:
        cells = [f"{n:,}", f"{time_python(n, args.preset):,.0f}"]
        cells.append(f"{time_xla(n, args.preset):,.0f}")
        cells += [f"{time_jax_env(b, n, args.preset):,.0f}" for b in args.jax_backends]
        print("| " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
