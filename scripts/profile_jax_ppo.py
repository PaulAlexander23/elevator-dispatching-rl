"""Where a JAX PPO update spends its time: env steps, policy, and the PPO epochs.

Each part is compiled on its own and timed per rollout of n_envs x n_steps:
- env: the vectorised env stepped n_steps times with fixed actions;
- policy: the policy forward pass and action sampling, n_steps times;
- rollout: both together, as `make_train` collects a rollout;
- update: one whole `make_train` update (rollout, GAE and the PPO epochs).
The PPO epochs and GAE take update - rollout. Preset full, relative obs,
256x256 MLPs, 3 epochs.

    uv run --group warp python scripts/profile_jax_ppo.py --backends jax warp
    uv run --group warp python scripts/profile_jax_ppo.py --trace /tmp/trace

`--trace` also records a `jax.profiler` trace of a few updates per setting,
for TensorBoard's profile plugin or Perfetto (ui.perfetto.dev).
"""

import argparse
import time

import jax
import jax.numpy as jnp

from elevator_rl.jax_env import JaxBuildingEnv
from elevator_rl.jax_ppo import PPOConfig, make_train, policy

SETTINGS = [(64, 32, 64), (1024, 8, 2048), (16384, 8, 16384)]


def seconds(fn, arg, repeats):
    """Mean wall time of fn(arg) over `repeats` calls, after one to compile."""
    jax.block_until_ready(fn(arg))
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(arg)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / repeats


def make_env(backend, preset="full"):
    if backend == "warp":
        from elevator_rl.warp_env import WarpBuildingEnv

        return WarpBuildingEnv(preset, reward_shaping=True)
    return JaxBuildingEnv(preset, reward_shaping=True)


def profile(backend, n_envs, n_steps, batch_size, repeats, trace=None):
    env = make_env(backend)
    config = PPOConfig(n_envs=n_envs, n_steps=n_steps, batch_size=batch_size, net_arch=(256, 256))
    init, update = make_train(env, config)
    runner = init(jax.random.key(0))
    n_lifts = env.config.n_lifts
    vec_reset, vec_step = env.make_vec_env(n_envs)
    actions = jnp.zeros((n_envs, n_lifts), jnp.int32)

    @jax.jit
    def env_only(vec_state):
        def body(vec_state, _):
            vec_state, obs, reward, done, _ = vec_step(vec_state, actions)
            return vec_state, reward.sum()

        return jax.lax.scan(body, vec_state, None, length=n_steps)

    @jax.jit
    def policy_only(args):
        params, obs, key = args

        def body(key, _):
            key, sub = jax.random.split(key)
            logits, value = policy(params, obs, n_lifts)
            return key, (jax.random.categorical(sub, logits), value)

        return jax.lax.scan(body, key, None, length=n_steps)

    @jax.jit
    def rollout(args):
        params, vec_state, obs, key = args

        def body(carry, _):
            vec_state, obs, key = carry
            key, sub = jax.random.split(key)
            logits, value = policy(params, obs, n_lifts)
            acts = jax.random.categorical(sub, logits)
            vec_state, obs, reward, done, info = vec_step(vec_state, acts)
            return (vec_state, obs, key), (reward, value)

        return jax.lax.scan(body, (vec_state, obs, key), None, length=n_steps)

    full = jax.jit(update)
    key = jax.random.key(1)
    vec_state, obs = vec_reset(key)
    times = {
        "env": seconds(env_only, vec_state, repeats),
        "policy": seconds(policy_only, (runner.params, obs, key), repeats),
        "rollout": seconds(rollout, (runner.params, vec_state, obs, key), repeats),
        "update": seconds(lambda r: full(r)[0], runner, repeats),
    }
    if trace:
        jax.block_until_ready(full(runner))
        with jax.profiler.trace(f"{trace}/{backend}-{n_envs}x{n_steps}-mb{batch_size}"):
            for _ in range(3):
                runner, metrics = full(runner)
            jax.block_until_ready(metrics)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backends", nargs="+", choices=("jax", "warp"), default=["jax", "warp"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trace", help="directory for jax.profiler traces")
    args = parser.parse_args()

    print(
        "| Env | Envs x steps : minibatch | Env | Policy | Rollout | Epochs + GAE | Update "
        "| PPO steps/s |"
    )
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    for backend in args.backends:
        for n_envs, n_steps, batch_size in SETTINGS:
            t = profile(backend, n_envs, n_steps, batch_size, args.repeats, args.trace)
            total = t["update"]
            epochs = total - t["rollout"]

            def cell(x, total=total):
                return f"{x * 1e3:.1f} ms ({x / total:.0%})"

            print(
                f"| {backend} | {n_envs}x{n_steps} : {batch_size} | {cell(t['env'])} | "
                f"{cell(t['policy'])} | {cell(t['rollout'])} | {cell(epochs)} | "
                f"{total * 1e3:.1f} ms | {n_envs * n_steps / total:,.0f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
