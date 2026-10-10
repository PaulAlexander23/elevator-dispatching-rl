"""The C API shared library (libelevator_c) through ctypes: CtypesVecEnv
against CppVecEnv, plus the C API's error handling.

Skipped unless both are built:

    cmake -S cpp -B cpp/build -DPython_EXECUTABLE=.venv/bin/python
    cmake --build cpp/build
"""

import ctypes
import threading

import numpy as np
import pytest

from elevator_rl import cpp_env, ctypes_env
from elevator_rl.building_env import BuildingEnv
from elevator_rl.ctypes_env import CtypesVecEnv

pytestmark = pytest.mark.skipif(
    not (ctypes_env.available() and cpp_env.available()),
    reason="libelevator_c or elevator_rl._cpp not built",
)

OPTIONS = [
    {"obs_type": "custom"},
    {"obs_type": "relative", "reward_shaping": True},
    {
        "obs_type": "box",
        "reward_shaping": "progress_waiting",
        "action_mode": "target",
        "observe_direction": True,
    },
]


def assert_infos_equal(a, b):
    for x, y in zip(a, b, strict=True):
        assert x.keys() == y.keys()
        if x:
            np.testing.assert_array_equal(x["terminal_observation"], y["terminal_observation"])
            assert x["TimeLimit.truncated"] == y["TimeLimit.truncated"]
            assert (x["episode"]["r"], x["episode"]["l"]) == (y["episode"]["r"], y["episode"]["l"])


@pytest.mark.parametrize("options", OPTIONS, ids=["custom", "relative", "target"])
@pytest.mark.parametrize("preset", ["original", "multi", "full"])
def test_matches_nanobind_step_for_step(preset, options):
    """Same seeds and actions give identical batches, across auto-resets."""
    kwargs = dict(n_envs=5, max_steps=40, seed=11, **options)
    c, nb = CtypesVecEnv(preset, **kwargs), cpp_env.CppVecEnv(preset, **kwargs)
    assert c.observation_space == nb.observation_space
    assert c.action_space == nb.action_space

    np.testing.assert_array_equal(c.reset(), nb.reset())
    rng = np.random.default_rng(0)
    nvec = c.action_space.nvec
    for step in range(150):
        if step == 90:  # reseed mid-run: both take the new seeds at reset
            c.seed(3)
            nb.seed(3)
            np.testing.assert_array_equal(c.reset(), nb.reset())
        actions = rng.integers(nvec, size=(5, len(nvec)))
        c_obs, c_rewards, c_dones, c_infos = c.step(actions)
        nb_obs, nb_rewards, nb_dones, nb_infos = nb.step(actions)
        np.testing.assert_array_equal(c_obs, nb_obs)
        np.testing.assert_array_equal(c_rewards, nb_rewards)
        np.testing.assert_array_equal(c_dones, nb_dones)
        assert_infos_equal(c_infos, nb_infos)
        assert c_obs.dtype == nb_obs.dtype and c_rewards.dtype == nb_rewards.dtype
    c.close()


def test_returned_arrays_are_copies():
    env = CtypesVecEnv("multi", n_envs=2)
    first = env.reset()
    kept = first.copy()
    env.step(np.zeros((2, 4), np.int64))
    np.testing.assert_array_equal(first, kept)


def test_out_of_range_action_raises_and_writes_nothing():
    env = CtypesVecEnv("multi", n_envs=3)
    env.reset()
    obs_before = env._obs.copy()
    actions = np.zeros((3, 4), np.int64)
    actions[2, 1] = env.action_space.nvec[1]
    with pytest.raises(RuntimeError, match="action out of range"):
        env.step(actions)
    np.testing.assert_array_equal(env._obs, obs_before)
    # The env is still usable.
    env.step(np.zeros((3, 4), np.int64))


def test_bad_arguments_become_errors_not_crashes():
    lib = ctypes_env._lib
    options = ctypes_env.Options()
    assert lib.elevator_default_options(ctypes.byref(options)) == 0
    assert (options.max_steps, options.gamma) == (200, 0.99)

    # A C++ exception thrown by parse_config comes back as NULL plus a message.
    assert not lib.elevator_vec_create(b"n_floors=10 colour=red", ctypes.byref(options), 1, 0, 0)
    assert b"unknown config key: colour" in lib.elevator_last_error()
    assert not lib.elevator_vec_create(b"n_lifts=99", ctypes.byref(options), 1, 0, 0)
    assert b"n_lifts out of range" in lib.elevator_last_error()
    assert not lib.elevator_vec_create(b"", ctypes.byref(options), 1, 7, 0)
    assert b"obs_type" in lib.elevator_last_error()
    assert not lib.elevator_vec_create(b"", ctypes.byref(options), 0, 0, 0)
    assert b"n_envs" in lib.elevator_last_error()
    assert not lib.elevator_vec_create(None, ctypes.byref(options), 1, 0, 0)

    assert lib.elevator_vec_num_envs(None) == -1
    assert lib.elevator_vec_reset(None, None, None) == 1
    lib.elevator_vec_destroy(None)


def test_errors_are_per_thread():
    lib = ctypes_env._lib
    options = ctypes_env.Options()
    lib.elevator_default_options(ctypes.byref(options))
    lib.elevator_vec_create(b"colour=red", ctypes.byref(options), 1, 0, 0)
    seen = []

    def other():
        lib.elevator_vec_create(b"n_lifts=99", ctypes.byref(options), 1, 0, 0)
        seen.append(lib.elevator_last_error())

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert b"n_lifts" in seen[0]
    assert b"colour" in lib.elevator_last_error()


def test_close_frees_the_handle_once():
    env = CtypesVecEnv("original", n_envs=2)
    assert env._free.alive
    env.close()
    assert not env._free.alive
    env.close()  # a second close is a no-op


def test_abi_checks():
    lib = ctypes_env._lib
    assert lib.elevator_abi_version() == ctypes_env.ABI_VERSION
    assert lib.elevator_options_size() == ctypes.sizeof(ctypes_env.Options) == 56


def test_handle_reports_sizes():
    reference = BuildingEnv("full", obs_type="relative")
    handle = ctypes_env.create(reference, 3, "relative")
    lib = ctypes_env._lib
    try:
        assert lib.elevator_vec_num_envs(handle) == 3
        assert lib.elevator_vec_n_lifts(handle) == reference.config.n_lifts
        assert lib.elevator_vec_n_actions(handle) == reference.action_space.nvec[0]
        assert lib.elevator_vec_observation_size(handle) == reference.observation_space.shape[0]
    finally:
        lib.elevator_vec_destroy(handle)


def test_ppo_trains_on_the_ctypes_env(tmp_path):
    from elevator_rl.train import main

    save_path = tmp_path / "model.zip"
    main(
        total_timesteps=256,
        eval_freq=256,
        n_eval_episodes=1,
        save_path=save_path,
        n_steps=16,
        n_envs=8,
        obs_type="relative",
        preset="full",
        vec_env="ctypes",
    )
    assert save_path.exists()
