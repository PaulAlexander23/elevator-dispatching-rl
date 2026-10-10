"""Stable-Baselines3 `VecEnv` backed by EnvPool running the C++ elevator env.

EnvPool is built from source with the env added (scripts/build_envpool.sh),
so `available()` says whether that wheel is installed. Compared with
`CppVecEnv`, which steps every env in one thread, EnvPool steps them on a pool
of C++ threads and hands back the batch.

EnvPool resets a finished env on the step *after* it finishes (it ignores
that step's action), while SB3 expects the reset on the same step, with the
last observation in `info["terminal_observation"]`. `EnvPoolVecEnv` bridges
the two by resetting the finished envs right away, in one batched call.
"""

import time

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from elevator_rl.building_env import BuildingEnv

try:
    import envpool
    import envpool.elevator  # noqa: F401  (the wheel was built with the env)
except ImportError:  # not installed
    envpool = None

OBS_TYPES = ("box", "relative")  # EnvPool's spec fixes the obs dtype to float32


def available():
    return envpool is not None


def _require():
    if envpool is None:
        raise ImportError(
            "envpool with the elevator env is not installed: run "
            "`scripts/build_envpool.sh` and install the wheel it builds (Python 3.12+)"
        )
    return envpool


def envpool_kwargs(env: BuildingEnv):
    """EnvPool config for a Python env's constructor arguments."""
    from elevator_rl.trace import config_line

    shaping = env.reward_shaping
    kwargs = {
        "building": config_line(env.config).removeprefix("config "),
        "obs_type": env.obs_type,
        "shaped": shaping is not None,
        "action_mode": env.action_mode,
        "observe_direction": env.observe_direction,
        "max_episode_steps": env.max_steps,
    }
    if shaping is not None:
        for name in ("pickup", "empty_serve", "progress", "waiting", "gamma"):
            kwargs[name] = getattr(shaping, name)
    return kwargs


class EnvPoolVecEnv(VecEnv):
    """`n_envs` C++ BuildingEnvs on EnvPool's thread pool, stepped as one batch.

    Takes the same arguments as `CppVecEnv`, plus `num_threads` (0: one per
    core, up to `n_envs`). Runs EnvPool in sync mode (batch size = n_envs),
    which is what an on-policy learner needs. Seeding follows EnvPool: env i
    gets the seed `seed + i`, and `seed(s)` rebuilds the pool at the next reset,
    so it restarts every env's seed stream.
    """

    def __init__(
        self,
        config=None,
        n_envs=1,
        obs_type="relative",
        reward_shaping=False,
        max_steps=200,
        action_mode="step",
        observe_direction=False,
        seed=0,
        num_threads=0,
    ):
        _require()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"EnvPool supports obs_type {OBS_TYPES}, got {obs_type!r}")
        # The spaces come from a Python env, as for CppVecEnv.
        self._reference = BuildingEnv(
            config,
            reward_shaping=reward_shaping,
            obs_type=obs_type,
            max_steps=max_steps,
            action_mode=action_mode,
            observe_direction=observe_direction,
        )
        super().__init__(n_envs, self._reference.observation_space, self._reference.action_space)
        self._kwargs = envpool_kwargs(self._reference)
        self.num_threads = num_threads
        self._pool_seed = seed
        self._pool = self._make_pool()
        self._returns = np.zeros(n_envs, np.float64)
        self._lengths = np.zeros(n_envs, np.int64)
        self._actions = None
        self._start = time.time()

    def _make_pool(self):
        return envpool.make(
            "Elevator-v0",
            env_type="gymnasium",
            num_envs=self.num_envs,
            batch_size=self.num_envs,
            num_threads=self.num_threads,
            seed=self._pool_seed,
            **self._kwargs,
        )

    def reset(self):
        seed = self._seeds[0]
        if seed is not None:
            # A new pool restarts every env's seed stream.
            self._pool.close()
            self._pool_seed = seed
            self._pool = self._make_pool()
        self._reset_seeds()
        self._reset_options()
        obs, _ = self._pool.reset()
        self._returns[:] = 0
        self._lengths[:] = 0
        return obs

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.int32).reshape(self.num_envs, -1)
        self._actions = np.ascontiguousarray(actions)

    def step_wait(self):
        obs, rewards, terminated, truncated, _ = self._pool.step(self._actions)
        dones = terminated | truncated
        self._returns += rewards
        self._lengths += 1
        infos = [{} for _ in range(self.num_envs)]
        finished = np.flatnonzero(dones)
        if finished.size:
            for i in finished:
                infos[i] = {
                    "terminal_observation": obs[i].copy(),
                    "TimeLimit.truncated": bool(truncated[i] and not terminated[i]),
                    "episode": {
                        "r": float(self._returns[i]),
                        "l": int(self._lengths[i]),
                        "t": round(time.time() - self._start, 6),
                    },
                }
            # Reset them now, rather than on EnvPool's next step.
            reset_obs, reset_info = self._pool.reset(finished.astype(np.int32))
            obs = obs.copy()
            obs[reset_info["env_id"]] = reset_obs
            self._returns[finished] = 0
            self._lengths[finished] = 0
        return obs, rewards.astype(np.float32), dones, infos

    def close(self):
        self._pool.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self._reference, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("EnvPoolVecEnv envs have no settable attributes")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("EnvPoolVecEnv envs have no Python methods")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
