"""Where a JAX env step spends its time, part by part.

Each part of `JaxBuildingEnv.step` is vmapped over the envs, compiled on its
own and timed: drawing and queueing the arrivals, applying the lifts'
actions (boarding and moves, plus the kinematics substeps), the substeps on
their own, the observation, then the whole step and the batched step with
auto-reset.
Random actions, preset full by default.

    uv run --group jax-cuda python scripts/profile_jax_env.py --n-envs 1024 16384
"""

import argparse
import time

import jax

from elevator_rl.jax_env import JaxBuildingEnv

REPEATS = 50


def seconds(fn, *args):
    jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / REPEATS


def profile(preset, obs_type, n_envs):
    env = JaxBuildingEnv(preset, reward_shaping=True, obs_type=obs_type)
    c = env.config
    reset, vec_step = env.make_vec_env(n_envs)
    vec_state, _ = reset(jax.random.key(0))
    # A mid-episode state: step a while with random actions first.
    keys = jax.random.split(jax.random.key(1), 60)
    step_jit = jax.jit(vec_step)
    for k in keys:
        actions = jax.random.randint(k, (n_envs, c.n_lifts), 0, env.n_actions)
        vec_state, *_ = step_jit(vec_state, actions)
    states = vec_state.env
    actions = jax.random.randint(jax.random.key(2), (n_envs, c.n_lifts), 0, env.n_actions)
    arrival_keys = jax.random.split(jax.random.key(3), n_envs)

    def arrivals(state, key):
        return env.add_arrivals(state, env.draw_arrivals(key, state.steps / env.max_steps))

    parts = {
        "arrivals": jax.jit(jax.vmap(arrivals)),
        "actions + substeps": jax.jit(jax.vmap(env.apply_actions)),
        "substeps only": jax.jit(
            jax.vmap(lambda s: jax.lax.fori_loop(0, c.substeps, lambda _, s: env._integrate(s), s))
        ),
        "observation": jax.jit(jax.vmap(env.observe)),
        "step": jax.jit(jax.vmap(env.step)),
    }
    args = {
        "arrivals": (states, arrival_keys),
        "actions + substeps": (states, actions),
        "substeps only": (states,),
        "observation": (states,),
        "step": (states, actions),
    }
    times = {name: seconds(fn, *args[name]) for name, fn in parts.items()}
    times["vec step (auto-reset)"] = seconds(step_jit, vec_state, actions)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="full")
    parser.add_argument("--obs-type", default="relative")
    parser.add_argument("--n-envs", type=int, nargs="+", default=[1024, 16384])
    args = parser.parse_args()
    print(f"devices: {jax.devices()}")
    for n in args.n_envs:
        t = profile(args.preset, args.obs_type, n)
        step = t["step"]
        print(f"\n{n:,} envs, {args.preset}, {args.obs_type} obs: {n / step:,.0f} env steps/s")
        print("| Part | ms per batched step | Share of step |")
        print("| --- | --: | --: |")
        for name, s in t.items():
            print(f"| {name} | {s * 1e3:.2f} | {s / step:.0%} |")


if __name__ == "__main__":
    main()
