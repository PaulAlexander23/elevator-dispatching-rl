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

With `--algo mappo` the lifts share one actor, each acting on its own block
of the "relative" observation (and a one-hot lift id), while the critic sees
the whole observation, as in MAPPO; `transfer` moves such an actor to a
building with a different number of lifts.

With `--pretrain N` it first imitates the collective heuristic on N samples
(`jax_imitate`), as the SB3 recipe for `full` does.
"""

import argparse
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from elevator_rl.building import PRESETS
from elevator_rl.jax_env import ACTION_MODES, OBS_TYPES, JaxBuildingEnv, one_hot

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
    # >0: a first layer of this width applied to each lift's block of the
    # "relative" observation with shared weights, before net_arch.
    lift_encoder: int = 0
    # "mappo": one actor shared by the lifts, each on its own block of the
    # "relative" observation (plus a one-hot lift id if lift_id), and the
    # critic on the whole observation. "ppo": one actor over everything.
    algo: str = "ppo"
    lift_id: bool = True


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
    """The MLP's output for x. A first layer narrower than x is a per-lift
    encoder: it is applied to each equal block of x (one per lift) with
    shared weights, and the blocks' outputs are concatenated."""
    first = layers[0]["w"].shape[0]
    if first != x.shape[-1]:
        lead = x.shape[:-1]
        blocks = x.reshape(-1, first)  # one row per lift: a plain 2-D matmul
        x = jnp.tanh(blocks @ layers[0]["w"] + layers[0]["b"]).reshape(*lead, -1)
        layers = layers[1:]
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def init_net(key, env, config: PPOConfig, n_out, out_gain):
    """An MLP over the env's features, with the per-lift encoder if configured."""
    if not config.lift_encoder:
        return init_mlp(key, feature_size(env), config.net_arch, n_out, out_gain)
    if env.obs_type != "relative":
        raise ValueError("lift_encoder needs the per-lift blocks of the relative observation")
    n_lifts = env.config.n_lifts
    k_enc, k_mlp = jax.random.split(key)
    encoder = _dense(k_enc, feature_size(env) // n_lifts, config.lift_encoder, np.sqrt(2))
    width = n_lifts * config.lift_encoder
    return [encoder, *init_mlp(k_mlp, width, config.net_arch, n_out, out_gain)]


def feature_size(env):
    """The network's input size: the one-hot width for the "custom" observation."""
    if env.obs_type == "custom":
        return int(env.observation_nvec.sum())
    return env.observation_size


def make_features(env):
    """obs -> network input; one-hot encodes "custom" as SB3 does for MultiDiscrete."""
    if env.obs_type == "custom":
        nvec = env.observation_nvec
        return lambda obs: one_hot(obs, nvec)
    return lambda obs: obs


def init_params(key, env: JaxBuildingEnv, config: PPOConfig):
    k_pi, k_vf = jax.random.split(key)
    n_lifts = env.config.n_lifts
    # SB3's gains: sqrt(2) for hidden layers, 0.01 for the policy head, 1 for the value head.
    vf = init_net(k_vf, env, config, 1, 1.0)
    if config.algo == "mappo":
        if env.obs_type != "relative":
            raise ValueError("mappo needs the per-lift blocks of the relative observation")
        n_in = feature_size(env) // n_lifts + (n_lifts if config.lift_id else 0)
        hidden = ((config.lift_encoder,) if config.lift_encoder else ()) + tuple(config.net_arch)
        return {"actor": init_mlp(k_pi, n_in, hidden, env.n_actions, 0.01), "vf": vf}
    if config.algo != "ppo":
        raise ValueError(f"unknown algo {config.algo!r}: ppo or mappo")
    return {"pi": init_net(k_pi, env, config, n_lifts * env.n_actions, 0.01), "vf": vf}


def lift_views(obs, n_lifts, lift_id):
    """Each lift's block of the observation, (..., n_lifts, block), with a
    one-hot lift id appended if lift_id."""
    views = obs.reshape(*obs.shape[:-1], n_lifts, -1)
    if not lift_id:
        return views
    ids = jnp.broadcast_to(jnp.eye(n_lifts, dtype=obs.dtype), (*obs.shape[:-1], n_lifts, n_lifts))
    return jnp.concatenate([views, ids], axis=-1)


def transfer(params, env, config: PPOConfig, key):
    """A mappo actor trained with one lift count, made ready for `env`'s.

    The actor's weights carry over; with lift ids, lifts beyond the trained
    count get a zero id weight (they look like no lift in particular) and
    surplus ids are dropped. The critic depends on the lift count, so it is
    fresh: fine for evaluation, to be trained by any fine-tuning.
    """
    if "actor" not in params:
        raise ValueError("only a mappo actor transfers across lift counts")
    fresh = init_params(key, env, config)
    actor = list(params["actor"])
    if config.lift_id:
        block = feature_size(env) // env.config.n_lifts
        w = actor[0]["w"]
        trained = w.shape[0] - block
        n_lifts = env.config.n_lifts
        ids = jnp.zeros((n_lifts, w.shape[1]), w.dtype)
        ids = ids.at[: min(trained, n_lifts)].set(w[block : block + min(trained, n_lifts)])
        actor[0] = {"w": jnp.concatenate([w[:block], ids]), "b": actor[0]["b"]}
    return {"actor": actor, "vf": fresh["vf"]}


def policy(params, obs, n_lifts):
    """Per-lift logits (..., n_lifts, n_actions) and the value (...)."""
    if "actor" in params:  # mappo: the shared actor on each lift's view
        block = obs.shape[-1] // n_lifts
        lift_id = params["actor"][0]["w"].shape[0] > block
        logits = mlp(params["actor"], lift_views(obs, n_lifts, lift_id))
    else:
        logits = mlp(params["pi"], obs).reshape(*obs.shape[:-1], n_lifts, -1)
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
    features = make_features(env)
    vec_reset, vec_step = env.make_vec_env(config.n_envs)
    optimizer = make_optimizer(config)

    def init(key, params=None):
        """`params` starts from given weights (e.g. a warm start) instead of fresh ones."""
        k_params, k_env, k_run = jax.random.split(key, 3)
        if params is None:
            params = init_params(k_params, env, config)
        vec_state, obs = vec_reset(k_env)
        return Runner(params, optimizer.init(params), vec_state, obs, k_run)

    def env_step(runner, _):
        params, opt_state, vec_state, obs, key = runner
        key, k_act = jax.random.split(key)
        logits, value = policy(params, features(obs), n_lifts)
        actions = jax.random.categorical(k_act, logits)
        vec_state, next_obs, reward, done, info = vec_step(vec_state, actions)
        # Episodes only end at the time limit, so bootstrap through it, as
        # SB3 does: add the discounted value of the observation it ended on.
        reward = reward + jax.lax.cond(
            done.any(),
            # where, not `* done`: terminal_obs is only defined where done (the
            # Warp env leaves the rest unwritten), and NaN * 0 is NaN.
            lambda: jnp.where(
                done, config.gamma * policy(params, features(info["terminal_obs"]), n_lifts)[1], 0.0
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
        logits, value = policy(params, features(obs), n_lifts)
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
        _, last_value = policy(runner.params, features(runner.obs), n_lifts)
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
    features = make_features(env)

    def evaluate(params, key):
        vec_state, obs = vec_reset(key)

        def body(carry, _):
            vec_state, obs = carry
            logits, _ = policy(params, features(obs), n_lifts)
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
    action_mode="step",
    callback=None,
    params=None,
):
    """Train and return (params, history); history rows are dicts per eval.

    `backend` picks the env: "jax" (jax_env) or "warp" (warp_env).

    Throughput excludes compilation (the first update) and evaluation.
    `callback(row)` runs after each eval; returning True stops training early
    (the hyperparameter search uses it to prune). `params` starts PPO from
    given weights, such as `jax_imitate.warm_start`'s.
    """
    config = config or PPOConfig()
    options = {"obs_type": obs_type, "action_mode": action_mode}
    if backend == "warp":
        from elevator_rl.warp_env import WarpBuildingEnv as env_cls
    else:
        env_cls = JaxBuildingEnv
    env = env_cls(preset, reward_shaping=reward_shaping, **options)
    eval_env = env_cls(preset, reward_shaping=False, **options)
    init, update = make_train(env, config)
    update = jax.jit(update, donate_argnums=0)
    evaluate = jax.jit(make_evaluate(eval_env, n_eval_episodes))
    key = jax.random.key(seed)
    if params is not None:  # `update` donates the runner's buffers; the caller keeps theirs
        params = jax.tree.map(jnp.copy, params)
    runner = init(key, params)
    per_update = config.n_envs * config.n_steps
    n_updates = max(total_timesteps // per_update, 1)
    eval_every = max(eval_freq // per_update, 1)
    history = []
    if params is not None:  # record where the warm start begins
        rewards = np.asarray(evaluate(runner.params, jax.random.fold_in(key, 0)))
        history.append(
            {
                "timesteps": 0,
                "seconds": 0.0,
                "steps_per_second": 0.0,
                "eval_mean": float(rewards.mean()),
                "eval_std": float(rewards.std()),
                "train_return": None,
            }
        )
        if verbose:
            print(f"start: eval {rewards.mean():6.1f} ± {rewards.std():4.1f}", flush=True)

    t0 = time.perf_counter()
    runner, metrics = update(runner)
    jax.block_until_ready(metrics)
    compile_seconds = time.perf_counter() - t0
    train_seconds = 0.0
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
    parser.add_argument("--obs-type", choices=OBS_TYPES, default="relative")
    parser.add_argument("--action-mode", choices=ACTION_MODES, default="step")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument(
        "--lift-encoder", type=int, default=0, help="width of a shared per-lift first layer"
    )
    parser.add_argument("--algo", choices=("ppo", "mappo"), default="ppo")
    parser.add_argument(
        "--no-lift-id", action="store_true", help="mappo: don't tell the actor which lift it is"
    )
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=BACKENDS, default="jax", help="env implementation")
    parser.add_argument(
        "--pretrain",
        type=int,
        default=0,
        help="imitate the collective heuristic on this many samples first",
    )
    args = parser.parse_args(argv)
    config = PPOConfig(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        net_arch=tuple(args.net_arch),
        lift_encoder=args.lift_encoder,
        algo=args.algo,
        lift_id=not args.no_lift_id,
    )
    print(f"devices: {jax.devices()}")
    params = None
    if args.pretrain:
        from elevator_rl.jax_imitate import warm_start

        env = JaxBuildingEnv(args.preset, obs_type=args.obs_type)
        params, stats = warm_start(env, args.preset, config, args.pretrain, seed=args.seed)
        print(f"imitation: mean log-likelihood {stats['log_likelihood']:.3f}")
    return train(
        args.preset,
        args.timesteps,
        config,
        args.obs_type,
        seed=args.seed,
        backend=args.backend,
        eval_freq=args.eval_freq,
        action_mode=args.action_mode,
        params=params,
    )


if __name__ == "__main__":
    main()
