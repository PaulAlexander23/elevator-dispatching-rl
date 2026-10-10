"""Imitation warm start for the JAX PPO (jax_imitate). Skipped unless JAX is installed."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from elevator_rl.jax_env import JaxBuildingEnv  # noqa: E402
from elevator_rl.jax_imitate import collect, pretrain, warm_start  # noqa: E402
from elevator_rl.jax_ppo import PPOConfig, init_params, train  # noqa: E402


def test_collect_is_cached(tmp_path):
    first = collect("multi", 300, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    second = collect("multi", 300, cache_dir=tmp_path)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)


def test_pretrain_fits_the_heuristic():
    env = JaxBuildingEnv("multi")
    config = PPOConfig(net_arch=(32, 32))
    observations, actions, returns = collect("multi", 2048)
    params = init_params(jax.random.key(0), env, config)
    _, before, _ = pretrain(params, observations, actions, returns, 4, epochs=1, lr=0.0)
    _, after, _ = pretrain(params, observations, actions, returns, 4, epochs=5)
    assert after > before + 1.0  # mean log-likelihood of the heuristic's 4-lift actions


def test_warm_start_feeds_ppo():
    env = JaxBuildingEnv("multi")
    config = PPOConfig(n_envs=8, n_steps=16, batch_size=64, net_arch=(16, 16))
    params, stats = warm_start(env, "multi", config, n_samples=512, epochs=2)
    assert np.isfinite(stats["value_loss"])
    _, history = train("multi", 256, config, params=params, eval_freq=128, verbose=False)
    assert history[0]["timesteps"] == 0  # the warm start's own eval comes first
    assert len(history) == 2  # the first update is compilation, so not evaluated
