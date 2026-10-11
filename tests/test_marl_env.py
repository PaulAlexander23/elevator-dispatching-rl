"""The multi-agent view of the building (elevator_rl.marl_env).

Skipped unless JAX is installed (the `jax` dependency group); the Warp test
also needs Warp.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from elevator_rl.jax_env import JaxBuildingEnv  # noqa: E402
from elevator_rl.marl_env import (  # noqa: E402
    MultiAgentEnv,
    Observation,
    ObservationGlobalState,
    StepType,
    make_vec_env,
)


def random_actions(key, env, shape):
    return jax.random.randint(key, (*shape, env.config.n_lifts), 0, env.n_actions)


def matches_spec(observation, spec, lead=()):
    for field, s in zip(observation, spec, strict=True):
        assert field.shape == (*lead, *s.shape)
        assert field.dtype == s.dtype


@pytest.mark.parametrize("action_mode", ["step", "target"])
def test_matches_the_single_agent_env_over_an_episode(action_mode):
    env = JaxBuildingEnv("full", reward_shaping=True, max_steps=30, action_mode=action_mode)
    marl = MultiAgentEnv(env)
    assert (marl.num_agents, marl.action_dim, marl.time_limit) == (4, env.n_actions, 30)
    key = jax.random.key(0)
    state, obs = env.reset(key)
    m_state, timestep = jax.jit(marl.reset)(key)
    step, m_step = jax.jit(env.step), jax.jit(marl.step)

    def check(timestep, obs, steps):
        view = timestep.observation
        matches_spec(view, marl.observation_spec())
        # The per-lift views, side by side, are the single-agent observation.
        np.testing.assert_array_equal(np.asarray(view.agents_view).ravel(), obs)
        for lift in range(4):
            np.testing.assert_array_equal(view.global_state[lift], obs)
        assert np.asarray(view.action_mask).all()
        np.testing.assert_array_equal(view.step_count, np.full(4, steps))

    assert int(timestep.step_type) == StepType.FIRST
    check(timestep, obs, 0)
    for t, k in enumerate(jax.random.split(jax.random.key(1), 30)):
        actions = random_actions(k, env, ())
        state, obs, reward, truncated, info = step(state, actions)
        m_state, timestep = m_step(m_state, actions)
        check(timestep, obs, t + 1)
        np.testing.assert_array_equal(timestep.reward, np.full(4, reward))
        np.testing.assert_array_equal(timestep.discount, np.ones(4))
        assert bool(timestep.last()) == bool(truncated) == (t == 29)
        assert int(timestep.extras["delivered"]) == int(info["delivered"])


def test_jit_and_vmap():
    marl = MultiAgentEnv(JaxBuildingEnv("full"), global_state=False)
    keys = jax.random.split(jax.random.key(0), 5)
    state, timestep = jax.jit(jax.vmap(marl.reset))(keys)
    assert isinstance(timestep.observation, Observation)
    actions = random_actions(jax.random.key(1), marl.env, (5,))
    state, timestep = jax.jit(jax.vmap(marl.step))(state, actions)
    matches_spec(timestep.observation, marl.observation_spec(), lead=(5,))
    assert timestep.reward.shape == (5, 4)
    assert marl.action_spec().shape == (4,)
    assert (np.asarray(timestep.step_type) == StepType.MID).all()


def vec_rollout(env, n_envs, steps):
    marl = make_vec_env(env, n_envs)
    reset, step = jax.jit(marl.reset), jax.jit(marl.step)
    vec_state, timestep = reset(jax.random.key(0))
    matches_spec(timestep.observation, marl.observation_spec(), lead=(n_envs,))
    rows = []
    for k in jax.random.split(jax.random.key(1), steps):
        vec_state, timestep = step(vec_state, random_actions(k, env, (n_envs,)))
        rows.append(timestep)
    return marl, rows


def check_auto_reset(marl, rows):
    n_envs = rows[0].reward.shape[0]
    for t, timestep in enumerate(rows):
        matches_spec(timestep.observation, marl.observation_spec(), lead=(n_envs,))
        last = (t + 1) % marl.time_limit == 0
        assert (np.asarray(timestep.step_type) == (StepType.LAST if last else StepType.MID)).all()
        # After the reset, the observation is the next episode's first.
        expected = (t + 1) % marl.time_limit
        assert (np.asarray(timestep.observation.step_count) == expected).all()
    terminal = rows[marl.time_limit - 1].extras["terminal_observation"]
    assert (np.asarray(terminal.step_count) == marl.time_limit).all()
    assert not np.array_equal(
        terminal.agents_view, rows[marl.time_limit - 1].observation.agents_view
    )


def test_vec_env_on_jax_matches_its_make_vec_env():
    env = JaxBuildingEnv("full", reward_shaping=True, max_steps=6)
    marl, rows = vec_rollout(env, 3, 8)
    check_auto_reset(marl, rows)
    reset, step = env.make_vec_env(3)
    vec_state, obs = reset(jax.random.key(0))
    for k, timestep in zip(jax.random.split(jax.random.key(1), 8), rows, strict=True):
        vec_state, obs, reward, done, info = step(vec_state, random_actions(k, env, (3,)))
        np.testing.assert_array_equal(timestep.observation.agents_view.reshape(3, -1), obs)
        np.testing.assert_array_equal(timestep.reward, jnp.repeat(reward[:, None], 4, axis=1))


def test_vec_env_on_warp():
    pytest.importorskip("warp")
    from elevator_rl.warp_env import WarpBuildingEnv

    marl, rows = vec_rollout(WarpBuildingEnv("full", reward_shaping=True, max_steps=6), 3, 8)
    check_auto_reset(marl, rows)
    assert isinstance(rows[0].observation, ObservationGlobalState)


def test_needs_the_relative_observation():
    with pytest.raises(ValueError, match="relative"):
        MultiAgentEnv(JaxBuildingEnv("full", obs_type="box"))
