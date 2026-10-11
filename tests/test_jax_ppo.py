"""PPO in JAX (elevator_rl.jax_ppo). Skipped unless JAX is installed."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from elevator_rl.jax_env import JaxBuildingEnv  # noqa: E402
from elevator_rl.jax_ppo import (  # noqa: E402
    PPOConfig,
    entropy,
    init_params,
    log_prob,
    make_train,
    mlp,
    policy,
    train,
)


def test_multidiscrete_log_prob_and_entropy():
    # Two lifts, three actions: uniform logits for lift 0, certain for lift 1.
    logits = jnp.array([[[0.0, 0.0, 0.0], [0.0, -1e9, -1e9]]])
    actions = jnp.array([[2, 0]])
    assert float(log_prob(logits, actions)[0]) == pytest.approx(np.log(1 / 3))
    assert float(entropy(logits)[0]) == pytest.approx(np.log(3))


def test_policy_shapes():
    env = JaxBuildingEnv("full")
    params = init_params(jax.random.key(0), env, PPOConfig())
    obs = jnp.zeros((5, env.observation_size))
    logits, value = policy(params, obs, env.config.n_lifts)
    assert logits.shape == (5, 4, env.n_actions)
    assert value.shape == (5,)


def test_update_runs_and_changes_the_policy():
    env = JaxBuildingEnv("multi", reward_shaping=True, max_steps=20)
    config = PPOConfig(n_envs=8, n_steps=10, batch_size=40)
    init, update = make_train(env, config)
    runner = init(jax.random.key(0))
    before = runner.params
    for _ in range(2):  # 20 steps: the second update ends an episode (bootstrapping)
        runner, metrics = jax.jit(update)(runner)
    assert np.isfinite(float(metrics["value_loss"]))
    assert int(metrics["episodes"]) == 8
    moved = jax.tree.map(lambda a, b: bool((a != b).any()), before, runner.params)
    assert all(jax.tree.leaves(moved))


@pytest.mark.parametrize(
    "options",
    [
        {"obs_type": "custom"},
        {"action_mode": "target"},
        {"obs_type": "custom", "action_mode": "target"},
    ],
    ids=["custom", "target", "custom-target"],
)
def test_update_runs_with_target_actions_and_custom_obs(options):
    env = JaxBuildingEnv("hall_calls", reward_shaping=True, max_steps=20, **options)
    config = PPOConfig(n_envs=4, n_steps=10, batch_size=20, net_arch=(16, 16))
    init, update = make_train(env, config)
    runner = init(jax.random.key(0))
    if env.obs_type == "custom":
        assert runner.params["pi"][0]["w"].shape[0] == env.observation_nvec.sum()
    assert runner.params["pi"][-1]["w"].shape[1] == env.config.n_lifts * env.n_actions
    runner, metrics = jax.jit(update)(runner)
    assert np.isfinite(float(metrics["value_loss"]))


def test_lift_encoder_shares_weights_across_lifts():
    env = JaxBuildingEnv("full")
    config = PPOConfig(net_arch=(32,), lift_encoder=8)
    params = init_params(jax.random.key(0), env, config)
    block = env.observation_size // env.config.n_lifts
    encoder, hidden, head = params["pi"]
    assert encoder["w"].shape == (block, 8)
    assert hidden["w"].shape == (4 * 8, 32)
    obs = jax.random.normal(jax.random.key(1), (3, env.observation_size))
    # Lift by lift: the same encoder on each block, outputs side by side.
    encoded = jnp.concatenate(
        [
            jnp.tanh(obs[:, i * block : (i + 1) * block] @ encoder["w"] + encoder["b"])
            for i in range(env.config.n_lifts)
        ],
        axis=-1,
    )
    expected = jnp.tanh(encoded @ hidden["w"] + hidden["b"]) @ head["w"] + head["b"]
    np.testing.assert_allclose(mlp(params["pi"], obs), expected, rtol=1e-5, atol=1e-6)
    logits, value = policy(params, obs, env.config.n_lifts)
    assert logits.shape == (3, 4, env.n_actions)
    assert value.shape == (3,)


def test_lift_encoder_trains_and_warm_starts():
    from elevator_rl.jax_imitate import warm_start

    env = JaxBuildingEnv("multi", reward_shaping=True, max_steps=20)
    config = PPOConfig(n_envs=8, n_steps=10, batch_size=40, net_arch=(16,), lift_encoder=8)
    params, stats = warm_start(env, "multi", config, n_samples=512, epochs=2)
    assert np.isfinite(stats["log_likelihood"])
    init, update = make_train(env, config)
    runner = init(jax.random.key(0), params)
    runner, metrics = jax.jit(update)(runner)
    assert np.isfinite(float(metrics["value_loss"]))


def test_lift_encoder_needs_the_relative_observation():
    with pytest.raises(ValueError, match="relative"):
        init_params(
            jax.random.key(0), JaxBuildingEnv("multi", obs_type="box"), PPOConfig(lift_encoder=8)
        )


def test_minibatch_must_divide_the_rollout():
    with pytest.raises(ValueError, match="minibatches"):
        make_train(JaxBuildingEnv("multi"), PPOConfig(n_envs=8, n_steps=10, batch_size=64))


@pytest.mark.slow
def test_ppo_learns_the_original_preset():
    # SB3 with the same settings reaches about 134 in 200k steps (the
    # collective heuristic delivers 136, a random policy about 7).
    _, history = train(
        "original", 200_000, PPOConfig(net_arch=(256, 256)), eval_freq=200_000, verbose=False
    )
    assert history[-1]["eval_mean"] > 100
