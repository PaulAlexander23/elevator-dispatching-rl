"""Benchmark the ctypes route against the nanobind module: call overhead and
batched stepping. PPO throughput is `elevator_rl.sweep --vec-envs cpp ctypes`.

Tables:
1. Cost per call of a trivial getter, and of one raw batch step of a single
   env, for nanobind and for ctypes with three ways of passing the arrays:
   pointers cached up front (what CtypesVecEnv does), `array.ctypes.data`
   per call, and `np.ctypeslib.ndpointer` argtypes, which also check dtype,
   shape and flags as nanobind does. One more row keeps the GIL (PyDLL), to
   show what releasing it costs.
2. Env steps/s of the full SB3 `VecEnv.step` (infos and copies included).

    uv run python scripts/time_ctypes.py
"""

import argparse
import ctypes
import timeit

import numpy as np

from elevator_rl import cpp_env, ctypes_env
from elevator_rl.building_env import BuildingEnv
from elevator_rl.cpp_env import CppVecEnv
from elevator_rl.ctypes_env import CtypesVecEnv


def per_call(function, number):
    """Best of 5 repeats, in µs per call."""
    return min(timeit.repeat(function, number=number, repeat=5)) / number * 1e6


def buffers(n_envs, n_lifts, size):
    return [
        np.zeros((n_envs, n_lifts), np.int64),  # actions: 0 is a valid action
        np.zeros((n_envs, size), np.float32),
        np.zeros(n_envs, np.float32),
        np.zeros(n_envs, np.uint8),
        np.zeros((n_envs, size), np.float32),
        np.zeros(n_envs, np.float32),
        np.zeros(n_envs, np.int32),
    ]


def call_overhead(preset, n_envs, number):
    lib = ctypes_env._lib
    reference = BuildingEnv(preset, obs_type="relative")
    nb = cpp_env._cpp.VecEnv(
        cpp_env.cpp_config(reference.config),
        cpp_env.cpp_options(reference),
        n_envs,
        cpp_env._cpp.ObsType.relative,
        0,
    )
    handle = ctypes.c_void_p(ctypes_env.create(reference, n_envs, "relative"))
    arrays = buffers(n_envs, reference.config.n_lifts, nb.observation_size)
    seeds = np.full(n_envs, -1, np.int64)
    nb.reset(seeds, arrays[1])
    lib.elevator_vec_reset(handle, seeds.ctypes.data, arrays[1].ctypes.data)

    cached = [ctypes.c_void_p(a.ctypes.data) for a in arrays]
    step = lib.elevator_vec_step
    # A second prototype of the same function, with checking argtypes.
    checked = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.c_void_p,
        *(
            np.ctypeslib.ndpointer(a.dtype, a.ndim, a.shape, ("C_CONTIGUOUS", "WRITEABLE"))
            for a in arrays
        ),
    )(("elevator_vec_step", lib))
    getter = lib.elevator_vec_observation_size
    # The same library through PyDLL, which keeps the GIL during the call.
    held = ctypes_env._declare(ctypes.PyDLL(str(ctypes_env.LIBRARY_PATH))).elevator_vec_step

    def per_call_pointers():
        return step(handle, *(a.ctypes.data for a in arrays))

    rows = [
        ("getter", "nanobind (`vec.observation_size`)", lambda: nb.observation_size),
        ("getter", "ctypes (`elevator_vec_observation_size`)", lambda: getter(handle)),
        ("step", "nanobind", lambda: nb.step(*arrays)),
        ("step", "ctypes, cached pointers", lambda: step(handle, *cached)),
        ("step", "ctypes, cached pointers, GIL held (`PyDLL`)", lambda: held(handle, *cached)),
        ("step", "ctypes, `array.ctypes.data` per call", per_call_pointers),
        ("step", "ctypes, `ndpointer` argtypes", lambda: checked(handle, *arrays)),
    ]
    print(f"\n{preset}, {n_envs} env(s), relative obs: µs per call, best of 5 × {number:,}\n")
    print("| Call | Route | µs per call |")
    print("|---|---|---:|")
    for kind, name, function in rows:
        print(f"| {kind} | {name} | {per_call(function, number):.3f} |", flush=True)
    lib.elevator_vec_destroy(handle)


def vec_env_rate(cls, preset, n_envs, steps):
    venv = cls(preset, n_envs, "relative")
    rng = np.random.default_rng(0)
    actions = rng.integers(venv.action_space.nvec, size=(64, n_envs, len(venv.action_space.nvec)))
    venv.reset()
    best = 0.0
    for _ in range(3):
        seconds = timeit.timeit(lambda: [venv.step(a) for a in actions], number=steps // 64)
        best = max(best, (steps // 64) * 64 * n_envs / seconds)
    venv.close()
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="full")
    parser.add_argument("--n-envs", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    parser.add_argument("--env-steps", type=int, default=1_000_000, help="per row and repeat")
    args = parser.parse_args()

    call_overhead("original", 1, 200_000)
    call_overhead(args.preset, 1, 100_000)

    print(f"\n{args.preset}, relative obs: SB3 VecEnv.step, env steps/s, best of 3\n")
    print("| n_envs | nanobind (CppVecEnv) | ctypes (CtypesVecEnv) | ctypes / nanobind |")
    print("|---:|---:|---:|---:|")
    for n_envs in args.n_envs:
        steps = max(args.env_steps // n_envs, 64)
        nb = vec_env_rate(CppVecEnv, args.preset, n_envs, steps)
        c = vec_env_rate(CtypesVecEnv, args.preset, n_envs, steps)
        print(f"| {n_envs} | {nb:,.0f} | {c:,.0f} | {c / nb:.2f}× |", flush=True)


if __name__ == "__main__":
    main()
