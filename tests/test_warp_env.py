"""The Warp port (elevator_rl.warp_env) against the Python BuildingEnv.

As tests/test_jax_env.py: Python draws the arrivals and the Warp kernels are
handed the same ones, in float64, so observations match exactly and rewards
to rounding. Skipped unless Warp (the `warp` dependency group) is installed;
runs on the GPU when there is one, else on Warp's CPU backend.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("warp")

import jax.numpy as jnp  # noqa: E402

from elevator_rl.building_env import BuildingEnv  # noqa: E402
from elevator_rl.jax_env import WARMUP_ROUNDS, Arrivals, arrivals_from_list  # noqa: E402
from elevator_rl.warp_env import WarpBuildingEnv  # noqa: E402


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


def batch(arrivals):
    """One env's Arrivals with a leading env axis of 1."""
    return Arrivals(*(jnp.asarray(field)[None] for field in arrivals))


@pytest.mark.parametrize("options", OPTIONS, ids=IDS)
@pytest.mark.parametrize("preset", ["original", "multi", "kinematic", "hall_calls", "full"])
def test_warp_matches_python_step_for_step(preset, options):
    env = BuildingEnv(preset, max_steps=80, **options)
    wenv = WarpBuildingEnv(preset, max_steps=80, **options)
    assert wenv.observation_size == env.observation_space.shape[0]
    assert wenv.n_actions == env.action_space.nvec[0]
    n = env.config.n_floors
    drawn = []
    draw = env.sim.draw_arrivals
    env.sim.draw_arrivals = lambda fraction: drawn.append(draw(fraction)) or drawn[-1]
    rng = np.random.default_rng(0)

    for episode in range(2):
        drawn.clear()
        obs, _ = env.reset(seed=episode)
        assert len(drawn) == WARMUP_ROUNDS
        rounds = [arrivals_from_list(pairs, n) for pairs in drawn]
        warmup = Arrivals(*(np.stack(field)[None] for field in zip(*rounds, strict=True)))
        state, wobs = wenv.reset(wenv.empty_state(1), jnp.zeros(1, jnp.int32), warmup)
        np.testing.assert_array_equal(obs, np.asarray(wobs)[0])
        for _ in range(env.max_steps):
            action = rng.integers(env.action_space.nvec[0], size=env.config.n_lifts)
            drawn.clear()
            obs, reward, _, truncated, info = env.step(action)
            state, wobs, wreward, wdone, winfo = wenv.step(
                state,
                jnp.asarray(action)[None],
                batch(arrivals_from_list(drawn[0], n)),
                auto_reset=False,
            )
            np.testing.assert_array_equal(obs, np.asarray(wobs)[0])
            assert float(wreward[0]) == pytest.approx(reward, rel=1e-12, abs=1e-12)
            assert bool(wdone[0]) == truncated
            assert int(winfo["delivered"][0]) == info["delivered"]
            assert int(winfo["boarded"][0]) == info["boarded"]
        assert int(state["dropped"][0]) == 0


def test_random_policy_matches_python_in_distribution():
    """With Warp's own random draws, returns match Python's statistically."""
    wenv = WarpBuildingEnv("full")
    n_envs, episodes = 64, 2
    reset, step = wenv.make_vec_env(n_envs)
    step = jax.jit(step)
    state, _ = jax.jit(reset)(jax.random.key(1))
    rng = np.random.default_rng(1)
    returns = []
    for _ in range(episodes * wenv.max_steps):
        actions = jnp.asarray(rng.integers(wenv.n_actions, size=(n_envs, 4)))
        state, _, _, done, info = step(state, actions)
        returns += list(np.asarray(info["episode_return"])[np.asarray(done)])
    assert len(returns) == n_envs * episodes

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
        np.std(returns) / np.sqrt(len(returns)),
        np.std(python_returns) / np.sqrt(len(python_returns)),
    )
    assert abs(np.mean(returns) - np.mean(python_returns)) < 4 * error


def test_auto_reset_keeps_the_terminal_obs():
    wenv = WarpBuildingEnv("multi", max_steps=5)
    reset, step = wenv.make_vec_env(3)
    state, obs = reset(jax.random.key(0))
    for t in range(1, 11):
        state, obs, _, done, info = step(state, jnp.zeros((3, 4), jnp.int32))
        assert bool(done.all()) == (t % 5 == 0)
        if t % 5 == 0:
            assert (np.asarray(info["episode_length"]) == 5).all()
            assert (np.asarray(state["steps"]) == 0).all()
            assert not np.array_equal(obs, info["terminal_obs"])


@pytest.mark.parametrize(
    "options",
    [{}, {"obs_type": "custom", "action_mode": "target"}],
    ids=["relative", "custom-target"],
)
def test_jax_ppo_trains_on_the_warp_env(x64, options):
    from elevator_rl.jax_ppo import PPOConfig, make_train

    jax.config.update("jax_enable_x64", False)  # training runs in float32
    env = WarpBuildingEnv("multi", reward_shaping=True, max_steps=20, **options)
    init, update = make_train(env, PPOConfig(n_envs=8, n_steps=10, batch_size=40))
    runner = init(jax.random.key(0))
    for _ in range(2):  # 20 steps: the second update ends an episode
        runner, metrics = jax.jit(update)(runner)
    assert np.isfinite(float(metrics["value_loss"]))
    assert int(metrics["episodes"]) == 8
