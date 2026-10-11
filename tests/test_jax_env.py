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
    one_hot,
)


@pytest.fixture(autouse=True)
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


OPTIONS = [
    {"obs_type": "relative", "reward_shaping": True},
    {"obs_type": "box", "reward_shaping": "progress_waiting", "observe_direction": True},
    {"obs_type": "custom", "reward_shaping": True, "observe_direction": True},
    {"obs_type": "relative", "reward_shaping": True, "action_mode": "target"},
    {"obs_type": "custom", "action_mode": "target"},
    {"obs_type": "box", "action_mode": "target", "observe_direction": True},
]
IDS = ["relative", "box", "custom", "relative-target", "custom-target", "box-target"]


def recorded(env):
    """Patch a Python env so each draw_arrivals result is kept in a list."""
    drawn = []
    draw = env.sim.draw_arrivals
    env.sim.draw_arrivals = lambda fraction: drawn.append(draw(fraction)) or drawn[-1]
    return drawn


@pytest.mark.parametrize("options", OPTIONS, ids=IDS)
@pytest.mark.parametrize("preset", ["original", "multi", "kinematic", "hall_calls", "full"])
def test_jax_matches_python_step_for_step(preset, options):
    env = BuildingEnv(preset, max_steps=80, **options)
    jenv = JaxBuildingEnv(preset, max_steps=80, **options)
    assert jenv.observation_size == env.observation_space.shape[0]
    assert jenv.n_actions == env.action_space.nvec[0]
    if options["obs_type"] == "custom":
        np.testing.assert_array_equal(jenv.observation_nvec, env.observation_space.nvec)
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


@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.85])
def test_arrivals_match_python_in_distribution(fraction):
    """JAX's arrival draw (a Poisson total split over floors) against Python's
    per-floor Poisson counts: count means and variances per floor, and the
    (origin, destination) histogram, all against Python's arrival model."""
    from scipy import stats

    jenv = JaxBuildingEnv("full")
    n, draws = jenv.config.n_floors, 20_000
    keys = jax.random.split(jax.random.key(0), draws)
    arrivals = jax.jit(jax.vmap(lambda k: jenv.draw_arrivals(k, fraction)))(keys)
    count = np.asarray(arrivals.count)
    assert not np.asarray(arrivals.overflow).any()

    env = BuildingEnv("full")
    env.reset(seed=0)
    rates, destinations = env.sim.arrival_model(fraction)
    expected = rates[:, None] * destinations  # mean arrivals per step, origin x destination
    np.testing.assert_allclose(np.asarray(jenv.arrival_model(fraction)[0]), rates, rtol=1e-12)

    # Per floor, mean and variance of the count are both the rate (Poisson).
    error = np.sqrt(rates / draws)
    assert np.all(np.abs(count.mean(axis=0) - rates) < 5 * error + 1e-12)
    np.testing.assert_allclose(count.var(axis=0), rates, rtol=0.1, atol=2e-3)

    def histogram(pairs):
        cells = np.zeros((n, n))
        np.add.at(cells, (pairs[:, 0], pairs[:, 1]), 1)
        return cells

    slot = np.arange(arrivals.destination.shape[-1])
    used = slot < count[..., None]  # (draws, floors, slots)
    origin = np.broadcast_to(np.arange(n)[None, :, None], used.shape)
    jax_pairs = np.stack([origin[used], np.asarray(arrivals.destination)[used]], axis=1)
    python_pairs = np.array(
        [pair for _ in range(draws) for pair in env.sim.draw_arrivals(fraction)]
    )
    # Pool the cells expected to see under 5 arrivals into one, then chi-square.
    big = expected * draws >= 5
    small = ~big & (expected > 0)
    assert expected[big].sum() / expected.sum() > 0.95
    for pairs in (jax_pairs, python_pairs):
        cells = histogram(pairs)
        assert cells[expected == 0].sum() == 0  # e.g. no trip to its own floor
        observed, mean = cells[big], expected[big] * draws
        if small.any():
            observed = np.append(observed, cells[small].sum())
            mean = np.append(mean, expected[small].sum() * draws)
        # Counts are Poisson, not multinomial: the total is free too.
        chi2 = ((observed - mean) ** 2 / mean).sum()
        assert stats.chi2.sf(chi2, df=len(mean)) > 1e-3


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


def test_one_hot_matches_sb3():
    import torch
    from stable_baselines3.common.preprocessing import preprocess_obs

    env = BuildingEnv("full", obs_type="custom", action_mode="target")
    obs = np.stack([env.reset(seed=s)[0] for s in range(3)])
    expected = preprocess_obs(torch.as_tensor(obs), env.observation_space).numpy()
    got = one_hot(jax.numpy.asarray(obs, jax.numpy.float32), env.observation_space.nvec)
    np.testing.assert_array_equal(got, expected)
