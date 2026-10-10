"""The elevator env on EnvPool (cpp/envpool/) and EnvPoolVecEnv.

Skipped unless the EnvPool wheel with the elevator env is installed:

    scripts/build_envpool.sh
    uv pip install cpp/build/envpool-dist/envpool-*.whl
"""

import numpy as np
import pytest

from elevator_rl import cpp_env, envpool_env
from elevator_rl.building_env import BuildingEnv

pytestmark = pytest.mark.skipif(
    not (envpool_env.available() and cpp_env.available()),
    reason="envpool with the elevator env, or elevator_rl._cpp, not installed",
)


def mt19937(seed):
    """Outputs of C++'s std::mt19937(seed), which EnvPool seeds each env with."""
    state = [seed & 0xFFFFFFFF]
    for i in range(1, 624):
        state.append((1812433253 * (state[-1] ^ (state[-1] >> 30)) + i) & 0xFFFFFFFF)
    index = 624
    while True:
        if index == 624:
            for i in range(624):
                y = (state[i] & 0x80000000) | (state[(i + 1) % 624] & 0x7FFFFFFF)
                state[i] = state[(i + 397) % 624] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
            index = 0
        y = state[index]
        index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        yield y ^ (y >> 18)


def test_mt19937_matches_the_standard():
    # The C++ standard fixes the 10000th output of a default-seeded mt19937.
    outputs = mt19937(5489)
    for _ in range(9999):
        next(outputs)
    assert next(outputs) == 4123659995


OPTIONS = [
    {"obs_type": "relative"},
    {
        "obs_type": "box",
        "reward_shaping": "progress_waiting",
        "action_mode": "target",
        "observe_direction": True,
    },
]


@pytest.mark.parametrize("options", OPTIONS, ids=["relative", "target"])
@pytest.mark.parametrize("num_threads", [1, 4])
def test_envpool_matches_the_cpp_env_step_for_step(options, num_threads):
    """Each pooled env replays exactly as a lone C++ env given its seeds."""
    n_envs, seed = 3, 11
    venv = envpool_env.EnvPoolVecEnv(
        "full", n_envs=n_envs, max_steps=20, seed=seed, num_threads=num_threads, **options
    )
    reference = BuildingEnv("full", max_steps=20, **options)
    obs_type = getattr(cpp_env._cpp.ObsType, options["obs_type"])
    singles = [cpp_env.make_cpp_env(reference) for _ in range(n_envs)]
    seeds = [mt19937(seed + i) for i in range(n_envs)]

    def expected_reset(i):
        singles[i].reset(next(seeds[i]))
        return singles[i].observe(obs_type)

    obs = venv.reset()
    for i in range(n_envs):
        np.testing.assert_array_equal(obs[i], expected_reset(i))

    rng = np.random.default_rng(0)
    for _ in range(50):  # two and a half episodes
        actions = rng.integers(venv.action_space.nvec[0], size=(n_envs, 4))
        obs, rewards, dones, infos = venv.step(actions)
        for i in range(n_envs):
            reward, truncated, _, _ = singles[i].step(actions[i].tolist())
            assert rewards[i] == np.float32(reward)
            assert dones[i] == truncated
            if truncated:
                np.testing.assert_array_equal(
                    infos[i]["terminal_observation"], singles[i].observe(obs_type)
                )
                np.testing.assert_array_equal(obs[i], expected_reset(i))
            else:
                np.testing.assert_array_equal(obs[i], singles[i].observe(obs_type))
    venv.close()


def test_spaces_and_episode_info():
    venv = envpool_env.EnvPoolVecEnv("full", n_envs=4, obs_type="relative", max_steps=5)
    reference = BuildingEnv("full", obs_type="relative")
    assert venv.observation_space == reference.observation_space
    assert venv.action_space == reference.action_space
    obs = venv.reset()
    assert obs.shape == (4, *reference.observation_space.shape)
    assert obs.dtype == np.float32
    venv.action_space.seed(0)
    for step in range(1, 11):
        _, rewards, dones, infos = venv.step(np.stack([venv.action_space.sample()] * 4))
        assert rewards.dtype == np.float32
        assert dones.all() == (step % 5 == 0)
        if dones.all():
            for info in infos:
                assert info["episode"]["l"] == 5
                assert info["TimeLimit.truncated"]
    venv.close()


def test_seed_rebuilds_the_pool():
    venv = envpool_env.EnvPoolVecEnv("multi", n_envs=2, obs_type="box")
    venv.seed(3)
    first = venv.reset()
    venv.seed(3)
    np.testing.assert_array_equal(first, venv.reset())
    venv.seed(4)
    assert not np.array_equal(first, venv.reset())
    venv.close()


def test_custom_obs_is_refused():
    with pytest.raises(ValueError, match="obs_type"):
        envpool_env.EnvPoolVecEnv("full", obs_type="custom")


def test_ppo_trains_on_envpool(tmp_path):
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
        vec_env="envpool",
    )
    assert save_path.exists()
