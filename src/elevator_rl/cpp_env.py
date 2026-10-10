"""Stable-Baselines3 `VecEnv` backed by the C++ port of `BuildingEnv`.

The extension module `elevator_rl._cpp` is built from cpp/ (see the README);
`available()` says whether it is. `CppVecEnv` steps every env in one call
into C++, which releases the GIL and writes into preallocated NumPy arrays,
so the per-step Python cost is one call per batch rather than one per env.
"""

import time

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from elevator_rl.building_env import BuildingEnv

try:
    from elevator_rl import _cpp
except ImportError:  # not built
    _cpp = None


def available():
    return _cpp is not None


def _require():
    if _cpp is None:
        raise ImportError(
            "elevator_rl._cpp is not built: run "
            "`cmake -S cpp -B cpp/build -DPython_EXECUTABLE=.venv/bin/python && "
            "cmake --build cpp/build`"
        )
    return _cpp


def cpp_config(config):
    """The C++ `Config` for a `BuildingConfig`."""
    from elevator_rl.trace import config_line

    return _require().parse_config(config_line(config).removeprefix("config "))


def cpp_options(env: BuildingEnv):
    """The C++ `EnvOptions` matching a Python env's constructor arguments."""
    cpp = _require()
    options = cpp.EnvOptions()
    options.shaped = env.reward_shaping is not None
    if env.reward_shaping is not None:
        for name in ("pickup", "empty_serve", "progress", "waiting", "gamma"):
            setattr(options.shaping, name, getattr(env.reward_shaping, name))
    options.max_steps = env.max_steps
    options.action_mode = getattr(cpp.ActionMode, env.action_mode)
    options.observe_direction = env.observe_direction
    return options


def make_cpp_env(env: BuildingEnv):
    """A single C++ env with the same settings as `env` (for parity checks)."""
    return _require().Env(cpp_config(env.config), cpp_options(env))


class CppVecEnv(VecEnv):
    """`n_envs` C++ BuildingEnvs, stepped as one batch.

    Takes the same arguments as `BuildingEnv`. Finished episodes reset
    automatically, with the last observation in
    `info["terminal_observation"]`, as `DummyVecEnv` does. Seeding follows the
    `VecEnv` convention: `seed(s)` gives env i the seed `s + i` at the next
    reset. Random draws differ from the Python env's, so trajectories match it
    only in distribution (the dynamics match exactly; see tests/test_cpp.py).
    """

    def __init__(
        self,
        config=None,
        n_envs=1,
        obs_type="custom",
        reward_shaping=False,
        max_steps=200,
        action_mode="step",
        observe_direction=False,
        seed=0,
    ):
        cpp = _require()
        # The spaces come from a Python env, so the two cannot disagree.
        self._reference = BuildingEnv(
            config,
            reward_shaping=reward_shaping,
            obs_type=obs_type,
            max_steps=max_steps,
            action_mode=action_mode,
            observe_direction=observe_direction,
        )
        super().__init__(n_envs, self._reference.observation_space, self._reference.action_space)
        self._vec = cpp.VecEnv(
            cpp_config(self._reference.config),
            cpp_options(self._reference),
            n_envs,
            getattr(cpp.ObsType, obs_type),
            seed,
        )
        n, size = n_envs, self._vec.observation_size
        dtype = np.int64 if obs_type == "custom" else np.float32
        self._obs = np.zeros((n, size), dtype)
        self._terminal = np.zeros((n, size), dtype)
        self._rewards = np.zeros(n, np.float32)
        self._dones = np.zeros(n, np.uint8)
        self._returns = np.zeros(n, np.float32)
        self._lengths = np.zeros(n, np.int32)
        self._actions = None
        self._start = time.time()

    def reset(self):
        seeds = np.array([-1 if s is None else s for s in self._seeds], dtype=np.int64)
        self._vec.reset(seeds, self._obs)
        self._reset_seeds()
        self._reset_options()
        return self._obs.copy()

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs, -1)
        self._actions = np.ascontiguousarray(actions)

    def step_wait(self):
        self._vec.step(
            self._actions,
            self._obs,
            self._rewards,
            self._dones,
            self._terminal,
            self._returns,
            self._lengths,
        )
        dones = self._dones.astype(bool)
        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(dones):
            infos[i] = {
                "terminal_observation": self._terminal[i].copy(),
                "TimeLimit.truncated": True,
                "episode": {
                    "r": float(self._returns[i]),
                    "l": int(self._lengths[i]),
                    "t": round(time.time() - self._start, 6),
                },
            }
        # Copies: SB3 keeps the previous observation while stepping.
        return self._obs.copy(), self._rewards.copy(), dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self._reference, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("CppVecEnv envs have no settable attributes")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("CppVecEnv envs have no Python methods")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
