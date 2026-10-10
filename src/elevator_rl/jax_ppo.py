"""PPO in JAX on the JAX env: rollout and update compiled into one program.

In the PureJaxRL style: `make_update` builds one function that collects a
rollout (a `lax.scan` over env steps, with the policy sampling on the
device) and then runs the PPO epochs over it, and `jax.jit` compiles the lot
into a single XLA program. Nothing crosses back to Python between steps, so
there is no per-step overhead at all: no VecEnv, no NumPy conversion, no
host-device copies.

The algorithm follows Stable-Baselines3's PPO so that the two can be
compared like for like: separate policy and value MLPs (tanh, orthogonal
initialisation with SB3's gains), a clipped surrogate loss, value MSE and an
entropy bonus, GAE, advantages normalised per minibatch, Adam (eps 1e-5)
with gradients clipped to norm 0.5, and at a time limit the reward is
topped up with the discounted value of the final observation, as SB3 does
for `TimeLimit.truncated`.

    uv run --group jax-cuda python -m elevator_rl.jax_ppo --preset full \\
        --n-envs 1024 --n-steps 8 --batch-size 2048 --timesteps 5000000
"""

import argparse
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from elevator_rl.building import PRESETS
from elevator_rl.jax_env import JaxBuildingEnv

BACKENDS = ("jax", "warp")


class PPOConfig(NamedTuple):
    n_envs: int = 64
    n_steps: int = 32  # per env, per rollout
    batch_size: int = 64  # minibatch
    n_epochs: int = 3  # as the rest of this project uses
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: tuple = (64, 64)  # SB3's MlpPolicy default


# -- network -------------------------------------------------------------------


def _dense(key, n_in, n_out, gain):
    weight = jax.nn.initializers.orthogonal(gain)(key, (n_in, n_out))
    return {"w": weight, "b": jnp.zeros(n_out)}


def init_mlp(key, n_in, hidden, n_out, out_gain):
    sizes = (n_in, *hidden)
    keys = jax.random.split(key, len(hidden) + 1)
    layers = [_dense(k, a, b, np.sqrt(2)) for k, a, b in zip(keys, sizes, sizes[1:], strict=False)]
    return layers + [_dense(keys[-1], sizes[-1], n_out, out_gain)]


def mlp(layers, x):
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def init_params(key, env: JaxBuildingEnv, config: PPOConfig):
    k_pi, k_vf = jax.random.split(key)
    n_logits = env.config.n_lifts * env.n_actions
    # SB3's gains: sqrt(2) for hidden layers, 0.01 for the policy head, 1 for the value head.
    return {
        "pi": init_mlp(k_pi, env.observation_size, config.net_arch, n_logits, 0.01),
        "vf": init_mlp(k_vf, env.observation_size, config.net_arch, 1, 1.0),
    }


def policy(params, obs, n_lifts):
    """Per-lift logits (..., n_lifts, n_actions) and the value (...)."""
    logits = mlp(params["pi"], obs)
    logits = logits.reshape(*obs.shape[:-1], n_lifts, -1)
    return logits, mlp(params["vf"], obs)[..., 0]


def log_prob(logits, actions):
    """Log-probability of a MultiDiscrete action: the sum over lifts."""
    log_p = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(log_p, actions[..., None], axis=-1)[..., 0].sum(axis=-1)


def entropy(logits):
    log_p = jax.nn.log_softmax(logits)
    return -(jnp.exp(log_p) * log_p).sum(axis=-1).sum(axis=-1)


# -- training ------------------------------------------------------------------


class Runner(NamedTuple):
    params: dict
    opt_state: optax.OptState
    vec_state: object
    obs: jax.Array
    key: jax.Array


class Rollout(NamedTuple):
    obs: jax.Array
    actions: jax.Array
    log_probs: jax.Array
    values: jax.Array
    rewards: jax.Array
    dones: jax.Array


def make_optimizer(config: PPOConfig):
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1e-5),
    )


def make_train(env: JaxBuildingEnv, config: PPOConfig):
    """(init, update): init(key) -> Runner; update(runner) -> (runner, metrics).

    `update` is one rollout of n_envs x n_steps plus the PPO epochs; jit it.
    """
    batch = config.n_envs * config.n_steps
    if batch % config.batch_size:
        raise ValueError(f"rollout {batch} does not divide into minibatches of {config.batch_size}")
    n_minibatches = batch // config.batch_size
    n_lifts = env.config.n_lifts
    vec_reset, vec_step = env.make_vec_env(config.n_envs)
    optimizer = make_optimizer(config)

    def init(key):
        k_params, k_env, k_run = jax.random.split(key, 3)
        params = init_params(k_params, env, config)
        vec_state, obs = vec_reset(k_env)
        return Runner(params, optimizer.init(params), vec_state, obs, k_run)

    def env_step(runner, _):
        params, opt_state, vec_state, obs, key = runner
        key, k_act = jax.random.split(key)
        logits, value = policy(params, obs, n_lifts)
        actions = jax.random.categorical(k_act, logits)
        vec_state, next_obs, reward, done, info = vec_step(vec_state, actions)
        # Episodes only end at the time limit, so bootstrap through it, as
        # SB3 does: add the discounted value of the observation it ended on.
        reward = reward + jax.lax.cond(
            done.any(),
            # where, not `* done`: terminal_obs is only defined where done (the
            # Warp env leaves the rest unwritten), and NaN * 0 is NaN.
            lambda: jnp.where(
                done, config.gamma * policy(params, info["terminal_obs"], n_lifts)[1], 0.0
            ),
            lambda: jnp.zeros_like(reward),
        )
        step = Rollout(obs, actions, log_prob(logits, actions), value, reward, done)
        metrics = {
            "episode_return": info["episode_return"],
            "episode_done": done,
        }
        return Runner(params, opt_state, vec_state, next_obs, key), (step, metrics)

    def advantages(rollout, last_value):
        def back(carry, step):
            gae, next_value = carry
            reward, value, done = step
            not_done = 1.0 - done
            delta = reward + config.gamma * next_value * not_done - value
            gae = delta + config.gamma * config.gae_lambda * not_done * gae
            return (gae, value), gae

        _, adv = jax.lax.scan(
            back,
            (jnp.zeros_like(last_value), last_value),
            (rollout.rewards, rollout.values, rollout.dones.astype(jnp.float32)),
            reverse=True,
        )
        return adv, adv + rollout.values

    def loss_fn(params, mb):
        obs, actions, old_log_probs, adv, returns = mb
        logits, value = policy(params, obs, n_lifts)
        ratio = jnp.exp(log_prob(logits, actions) - old_log_probs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        clipped = jnp.clip(ratio, 1 - config.clip_range, 1 + config.clip_range)
        policy_loss = -jnp.minimum(ratio * adv, clipped * adv).mean()
        value_loss = ((returns - value) ** 2).mean()
        entropy_loss = -entropy(logits).mean()
        loss = policy_loss + config.ent_coef * entropy_loss + config.vf_coef * value_loss
        clip_fraction = (jnp.abs(ratio - 1) > config.clip_range).mean()
        return loss, (policy_loss, value_loss, clip_fraction)

    def update(runner):
        runner, (rollout, ep) = jax.lax.scan(env_step, runner, None, length=config.n_steps)
        _, last_value = policy(runner.params, runner.obs, n_lifts)
        adv, returns = advantages(rollout, last_value)
        flat = jax.tree.map(
            lambda x: x.reshape(batch, *x.shape[2:]),
            (rollout.obs, rollout.actions, rollout.log_probs, adv, returns),
        )

        def epoch(carry, key):
            params, opt_state = carry
            order = jax.random.permutation(key, batch)
            minibatches = jax.tree.map(
                lambda x: x[order].reshape(n_minibatches, config.batch_size, *x.shape[1:]), flat
            )

            def minibatch(carry, mb):
                params, opt_state = carry
                grads, aux = jax.grad(loss_fn, has_aux=True)(params, mb)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), aux

            return jax.lax.scan(minibatch, (params, opt_state), minibatches)

        key, k_epochs = jax.random.split(runner.key)
        (params, opt_state), (pg, vf, clip) = jax.lax.scan(
            epoch, (runner.params, runner.opt_state), jax.random.split(k_epochs, config.n_epochs)
        )
        done = ep["episode_done"]
        n_done = done.sum()
        metrics = {
            "policy_loss": pg.mean(),
            "value_loss": vf.mean(),
            "clip_fraction": clip.mean(),
            "episodes": n_done,
            "mean_return": jnp.where(
                n_done > 0, jnp.where(done, ep["episode_return"], 0.0).sum() / n_done, 0.0
            ),
        }
        return runner._replace(params=params, opt_state=opt_state, key=key), metrics

    return init, update


def make_evaluate(env: JaxBuildingEnv, n_episodes):
    """evaluate(params, key) -> passengers delivered per episode, deterministic policy.

    `env` should be unshaped, so the return is the delivered count, as
    train.make_eval_env.
    """
    vec_reset, vec_step = env.make_vec_env(n_episodes)
    n_lifts = env.config.n_lifts

    def evaluate(params, key):
        vec_state, obs = vec_reset(key)

        def body(carry, _):
            vec_state, obs = carry
            logits, _ = policy(params, obs, n_lifts)
            vec_state, obs, reward, _, _ = vec_step(vec_state, logits.argmax(axis=-1))
            return (vec_state, obs), reward

        _, rewards = jax.lax.scan(body, (vec_state, obs), None, length=env.max_steps)
        return rewards.sum(axis=0)

    return evaluate


def train(
    preset="full",
    total_timesteps=1_000_000,
    config=None,
    obs_type="relative",
    reward_shaping=True,
    seed=0,
    eval_freq=100_000,
    n_eval_episodes=10,
    verbose=True,
    backend="jax",
    callback=None,
):
    """Train and return (params, history); history rows are dicts per eval.

    `backend` picks the env: "jax" (jax_env) or "warp" (warp_env).
    `callback(row)` runs after each eval; returning True stops training early
    (the hyperparameter search uses it to prune).

    Throughput excludes compilation (the first update) and evaluation.
    """
    config = config or PPOConfig()
    if backend == "warp":
        from elevator_rl.warp_env import WarpBuildingEnv as env_cls
    else:
        env_cls = JaxBuildingEnv
    env = env_cls(preset, reward_shaping=reward_shaping, obs_type=obs_type)
    eval_env = env_cls(preset, reward_shaping=False, obs_type=obs_type)
    init, update = make_train(env, config)
    update = jax.jit(update, donate_argnums=0)
    evaluate = jax.jit(make_evaluate(eval_env, n_eval_episodes))
    key = jax.random.key(seed)
    runner = init(key)
    per_update = config.n_envs * config.n_steps
    n_updates = max(total_timesteps // per_update, 1)
    eval_every = max(eval_freq // per_update, 1)

    t0 = time.perf_counter()
    runner, metrics = update(runner)
    jax.block_until_ready(metrics)
    compile_seconds = time.perf_counter() - t0
    train_seconds, history = 0.0, []
    for i in range(1, n_updates):
        t0 = time.perf_counter()
        runner, metrics = update(runner)
        if (i + 1) % eval_every == 0 or i == n_updates - 1:
            jax.block_until_ready(metrics)
            train_seconds += time.perf_counter() - t0
            rewards = np.asarray(evaluate(runner.params, jax.random.fold_in(key, i)))
            steps = (i + 1) * per_update
            row = {
                "timesteps": steps,
                "seconds": train_seconds,
                "steps_per_second": (steps - per_update) / train_seconds,
                "eval_mean": float(rewards.mean()),
                "eval_std": float(rewards.std()),
                "train_return": float(metrics["mean_return"]),
            }
            history.append(row)
            if verbose:
                print(
                    f"{steps:>10,d} steps  {row['seconds']:6.1f}s  "
                    f"{row['steps_per_second']:>9,.0f} steps/s  "
                    f"eval {row['eval_mean']:6.1f} ± {row['eval_std']:4.1f}",
                    flush=True,
                )
            if callback is not None and callback(row):
                break
        else:
            train_seconds += time.perf_counter() - t0
    if verbose:
        print(f"compile + first update: {compile_seconds:.1f}s")
    return runner.params, history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=PRESETS, default="full")
    parser.add_argument("--obs-type", choices=("box", "relative"), default="relative")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=BACKENDS, default="jax", help="env implementation")
    args = parser.parse_args(argv)
    config = PPOConfig(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        net_arch=tuple(args.net_arch),
    )
    print(f"devices: {jax.devices()}")
    return train(
        args.preset,
        args.timesteps,
        config,
        args.obs_type,
        seed=args.seed,
        backend=args.backend,
        eval_freq=args.eval_freq,
    )


if __name__ == "__main__":
    main()
