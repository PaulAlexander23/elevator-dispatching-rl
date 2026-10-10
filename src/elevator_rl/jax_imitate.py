"""Warm-start the JAX PPO policy by imitating the collective-control heuristic.

The JAX counterpart of `imitate.py`: the heuristic still runs on the Python
env (it is plain Python and only needs to run once), and the cloning is a
small jitted optax loop over the JAX PPO's own networks. The policy MLP is
fitted to the heuristic's actions (cross-entropy) and the value MLP to its
discounted returns, as `imitate.pretrain` does for SB3.

The JAX env's observations match the Python env's exactly (tests/test_jax_env),
so a policy cloned on Python observations runs unchanged on the JAX or Warp env.

    params, stats = warm_start(env, "full", config, n_samples=100_000)
    jax_ppo.train(..., params=params)
"""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from elevator_rl.jax_ppo import PPOConfig, init_params, log_prob, policy


def collect(preset, n_samples, gamma=0.99, seed=0, obs_type="relative", cache_dir=None):
    """`imitate.collect` with the default shaping, cached on disk when `cache_dir` is set.

    Collection runs the Python env (about a minute per 100k samples), so a
    sweep that warm-starts every run reuses one dataset.
    """
    from elevator_rl.imitate import collect as collect_python

    args = (preset, n_samples, "default", gamma, seed, obs_type)
    if cache_dir is None:
        return collect_python(*args)
    key = hashlib.sha1(json.dumps(args).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"imitation-{preset}-{n_samples}-{key}.npz"
    if path.exists():
        data = np.load(path)
        return data["observations"], data["actions"], data["returns"]
    observations, actions, returns = collect_python(*args)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observations=observations, actions=actions, returns=returns)
    return observations, actions, returns


def pretrain(params, observations, actions, returns, n_lifts, epochs=15, batch_size=256, lr=1e-3,
             seed=0):  # fmt: skip
    """Fit `params` to the data; return (params, final epoch's log-likelihood, value loss)."""
    optimizer = optax.adam(lr)
    n_batches = len(actions) // batch_size
    data = (
        jnp.asarray(observations, jnp.float32),
        jnp.asarray(actions, jnp.int32),
        jnp.asarray(returns, jnp.float32),
    )

    def loss_fn(params, batch):
        obs, act, ret = batch
        logits, value = policy(params, obs, n_lifts)
        log_likelihood = log_prob(logits, act).mean()
        value_loss = ((value - ret) ** 2).mean()
        return -log_likelihood + 0.5 * value_loss, (log_likelihood, value_loss)

    @jax.jit
    def epoch(carry, key):
        params, opt_state = carry
        order = jax.random.permutation(key, len(data[1]))[: n_batches * batch_size]
        batches = jax.tree.map(
            lambda x: x[order].reshape(n_batches, batch_size, *x.shape[1:]), data
        )

        def step(carry, batch):
            params, opt_state = carry
            grads, aux = jax.grad(loss_fn, has_aux=True)(params, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            return (optax.apply_updates(params, updates), opt_state), aux

        (params, opt_state), (ll, vl) = jax.lax.scan(step, (params, opt_state), batches)
        return (params, opt_state), (ll.mean(), vl.mean())

    carry = (params, optimizer.init(params))
    for key in jax.random.split(jax.random.key(seed), epochs):
        carry, (log_likelihood, value_loss) = epoch(carry, key)
    return carry[0], float(log_likelihood), float(value_loss)


def warm_start(
    env, preset, config: PPOConfig, n_samples=100_000, epochs=15, seed=0, cache_dir=None
):
    """Fresh PPO params for `env` (built from `preset`), cloned from the heuristic.

    Returns (params, {"log_likelihood", "value_loss"}).
    """
    observations, actions, returns = collect(
        preset, n_samples, config.gamma, seed, env.obs_type, cache_dir
    )
    params = init_params(jax.random.key(seed), env, config)
    params, log_likelihood, value_loss = pretrain(
        params, observations, actions, returns, env.config.n_lifts, epochs=epochs, seed=seed
    )
    return params, {"log_likelihood": log_likelihood, "value_loss": value_loss}
