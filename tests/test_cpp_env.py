"""The Python binding of the C++ port (elevator_rl._cpp) and CppVecEnv.

Skipped unless the module is built:

    cmake -S cpp -B cpp/build -DPython_EXECUTABLE=.venv/bin/python
    cmake --build cpp/build
"""

import numpy as np
import pytest

from elevator_rl import cpp_env
from elevator_rl.building import PRESETS
from elevator_rl.building_env import BuildingEnv

pytestmark = pytest.mark.skipif(not cpp_env.available(), reason="elevator_rl._cpp not built")

OPTIONS = [
    {},
    {"reward_shaping": True, "obs_type": "relative"},
    {
        "reward_shaping": "progress_waiting",
        "action_mode": "target",
        "observe_direction": True,
        "obs_type": "box",
    },
]


@pytest.mark.parametrize("options", OPTIONS, ids=["plain", "relative", "target"])
@pytest.mark.parametrize("preset", ["original", "multi", "full"])
def test_binding_matches_python_step_for_step(preset, options):
    """Python draws the arrivals; the C++ env is handed the same ones."""
    env = BuildingEnv(preset, max_steps=60, **options)
    drawn = []
    draw = env.sim.draw_arrivals
    env.sim.draw_arrivals = lambda fraction: drawn.append(draw(fraction)) or drawn[-1]
    cpp = cpp_env.make_cpp_env(env)
    obs_type = getattr(cpp_env._cpp.ObsType, env.obs_type)
    rng = np.random.default_rng(0)

    def reset(seed):
        drawn.clear()
        obs, _ = env.reset(seed=seed)
        cpp.reset(0, drawn)
        drawn.clear()
        np.testing.assert_array_equal(obs, cpp.observe(obs_type))

    reset(1)
    for _ in range(200):
        action = rng.integers(env.action_space.nvec[0], size=env.config.n_lifts)
        obs, reward, _, truncated, info = env.step(action)
        cpp_reward, cpp_truncated, delivered, boarded = cpp.step(action.tolist(), drawn.pop())
        assert (cpp_reward, cpp_truncated) == (reward, truncated)
        assert (delivered, boarded) == (info["delivered"], info["boarded"])
        np.testing.assert_array_equal(obs, cpp.observe(obs_type))
        if truncated:
            reset(int(rng.integers(1000)))


@pytest.mark.parametrize("obs_type", ["custom", "box", "relative"])
def test_vec_env_spaces_and_auto_reset(obs_type):
    env = cpp_env.CppVecEnv("full", n_envs=3, obs_type=obs_type, max_steps=5)
    reference = BuildingEnv("full", obs_type=obs_type)
    assert env.observation_space == reference.observation_space
    assert env.action_space == reference.action_space

    obs = env.reset()
    assert obs.shape == (3, *reference.observation_space.shape)
    env.action_space.seed(0)
    for step in range(1, 11):
        obs, rewards, dones, infos = env.step(np.stack([env.action_space.sample()] * 3))
        assert rewards.dtype == np.float32 and dones.dtype == bool
        assert all(env.observation_space.contains(o) for o in obs)
        assert dones.all() == (step % 5 == 0)
        if dones.all():
            for info in infos:
                assert info["episode"]["l"] == 5
                assert info["TimeLimit.truncated"]
                assert env.observation_space.contains(info["terminal_observation"])


def rollout(env, n_steps=50):
    env.action_space.seed(3)
    observations = [env.reset()]
    for _ in range(n_steps):
        actions = np.stack([env.action_space.sample() for _ in range(env.num_envs)])
        observations.append(env.step(actions)[0])
    return np.array(observations)


def test_vec_env_seeding_is_reproducible():
    a = cpp_env.CppVecEnv("full", n_envs=4)
    b = cpp_env.CppVecEnv("full", n_envs=4)
    a.seed(7)
    b.seed(7)
    first = rollout(a)
    np.testing.assert_array_equal(first, rollout(b))
    b.seed(8)
    assert not np.array_equal(first, rollout(b))
    # Envs in one batch get different seeds.
    assert not np.array_equal(first[:, 0], first[:, 1])


def test_vec_env_random_policy_matches_python_in_distribution():
    n_envs, episodes = 16, 6
    cpp = cpp_env.CppVecEnv("multi", n_envs=n_envs)
    cpp.seed(0)
    cpp.reset()
    cpp.action_space.seed(0)
    cpp_returns = []
    while len(cpp_returns) < n_envs * episodes:
        actions = np.stack([cpp.action_space.sample() for _ in range(n_envs)])
        _, _, dones, infos = cpp.step(actions)
        cpp_returns += [infos[i]["episode"]["r"] for i in np.flatnonzero(dones)]

    env = BuildingEnv("multi")
    env.action_space.seed(0)
    python_returns = []
    for episode in range(60):
        env.reset(seed=episode)
        total, truncated = 0.0, False
        while not truncated:
            _, reward, _, truncated, _ = env.step(env.action_space.sample())
            total += reward
        python_returns.append(total)

    error = np.hypot(
        np.std(cpp_returns) / np.sqrt(len(cpp_returns)),
        np.std(python_returns) / np.sqrt(len(python_returns)),
    )
    assert abs(np.mean(cpp_returns) - np.mean(python_returns)) < 4 * error


def test_ppo_trains_on_the_cpp_env(tmp_path):
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
        vec_env="cpp",
        pretrain=200,
    )
    assert save_path.exists()


def test_presets_build_in_cpp():
    for name, config in PRESETS.items():
        assert str(cpp_env.cpp_config(config)) == str(cpp_env._cpp.preset(name))
