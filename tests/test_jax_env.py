"""The JAX port (elevator_rl.jax_env) against the Python BuildingEnv.

Python draws the arrivals and the JAX env is handed the same ones, in float64
(jax_enable_x64), so observations match exactly. Rewards match to rounding:
the JAX env sums the progress potential per destination floor rather than
per passenger. Skipped unless JAX is installed (the `jax` dependency group).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from elevator_rl.building_env import BuildingEnv  # noqa: E402
from elevator_rl.jax_env import (  # noqa: E402
    WARMUP_ROUNDS,
    Arrivals,
    JaxBuildingEnv,
    arrivals_from_list,
    make_vec_env,
)


@pytest.fixture(autouse=True)
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


OPTIONS = [
    {"obs_type": "relative", "reward_shaping": True},
    {"obs_type": "box", "reward_shaping": "progress_waiting", "observe_direction": True},
]


def recorded(env):
    """Patch a Python env so each draw_arrivals result is kept in a list."""
    drawn = []
    draw = env.sim.draw_arrivals
    env.sim.draw_arrivals = lambda fraction: drawn.append(draw(fraction)) or drawn[-1]
    return drawn


@pytest.mark.parametrize("options", OPTIONS, ids=["relative", "box"])
@pytest.mark.parametrize("preset", ["original", "multi", "kinematic", "hall_calls", "full"])
def test_jax_matches_python_step_for_step(preset, options):
    env = BuildingEnv(preset, max_steps=80, **options)
    jenv = JaxBuildingEnv(preset, max_steps=80, **options)
    assert jenv.observation_size == env.observation_space.shape[0]
    n = env.config.n_floors
    drawn = recorded(env)
    step = jax.jit(jenv.step)
    reset = jax.jit(jenv.reset)
    rng = np.random.default_rng(0)

    for episode in range(2):
        drawn.clear()
        obs, _ = env.reset(seed=episode)
        assert len(drawn) == WARMUP_ROUNDS
        rounds = [arrivals_from_list(pairs, n) for pairs in drawn]
        warmup = Arrivals(*(np.stack(field) for field in zip(*rounds, strict=True)))
        state, jobs = reset(jax.random.key(0), warmup)
        np.testing.assert_array_equal(obs, jobs)
        for _ in range(env.max_steps):
            action = rng.integers(env.action_space.nvec[0], size=env.config.n_lifts)
            drawn.clear()
            obs, reward, _, truncated, info = env.step(action)
            state, jobs, jreward, jtruncated, jinfo = step(
                state, action, arrivals_from_list(drawn[0], n)
            )
            np.testing.assert_array_equal(obs, jobs)
            assert float(jreward) == pytest.approx(reward, rel=1e-12, abs=1e-12)
            assert bool(jtruncated) == truncated
            assert int(jinfo["delivered"]) == info["delivered"]
            assert int(jinfo["boarded"]) == info["boarded"]
        assert int(state.dropped) == 0


def test_random_policy_matches_python_in_distribution():
    """With JAX's own random draws, returns match Python's statistically."""
    jenv = JaxBuildingEnv("full")
    n_envs, episodes = 64, 2
    reset, step = make_vec_env(jenv, n_envs)

    def run(key):
        vec_state, _ = reset(key)

        def body(carry, key):
            vec_state = carry
            actions = jax.random.randint(key, (n_envs, 4), 0, jenv.n_actions)
            vec_state, _, _, done, info = step(vec_state, actions)
            return vec_state, (done, info["episode_return"])

        keys = jax.random.split(key, episodes * jenv.max_steps)
        _, (done, returns) = jax.lax.scan(body, vec_state, keys)
        return done, returns

    done, returns = jax.jit(run)(jax.random.key(1))
    jax_returns = np.asarray(returns)[np.asarray(done)]  # shapes vary: outside jit
    assert jax_returns.size == n_envs * episodes

    env = BuildingEnv("full")
    env.action_space.seed(0)
    python_returns = []
    for episode in range(40):
        env.reset(seed=episode)
        total, truncated = 0.0, False
        while not truncated:
            _, reward, _, truncated, _ = env.step(env.action_space.sample())
            total += reward
        python_returns.append(total)

    error = np.hypot(
        np.std(jax_returns) / np.sqrt(len(jax_returns)),
        np.std(python_returns) / np.sqrt(len(python_returns)),
    )
    assert abs(np.mean(jax_returns) - np.mean(python_returns)) < 4 * error


def test_vec_env_auto_resets_with_terminal_obs():
    jenv = JaxBuildingEnv("multi", max_steps=5)
    reset, step = make_vec_env(jenv, 3)
    vec_state, obs = reset(jax.random.key(0))
    step = jax.jit(step)
    for t in range(1, 11):
        actions = np.zeros((3, 4), np.int32)
        vec_state, obs, _, done, info = step(vec_state, actions)
        assert bool(done.all()) == (t % 5 == 0)
        if t % 5 == 0:
            assert (np.asarray(info["episode_length"]) == 5).all()
            assert (np.asarray(vec_state.env.steps) == 0).all()
            assert not np.array_equal(obs, info["terminal_obs"])
