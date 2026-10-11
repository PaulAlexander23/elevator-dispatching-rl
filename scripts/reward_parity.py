"""Reward parity across env backends: the same PPO, trained on each port.

Every SB3 VecEnv backend (Python, the C++ port through nanobind and ctypes,
EnvPool) trains SB3's PPO with the same settings and seeds, then is
evaluated on the Python env, so a port that changed the dynamics would show
up as a different reward. The JAX PPO on the JAX and Warp envs is added for
reference: a different trainer (matching SB3's algorithm), so it should
agree within noise but not seed for seed.

    uv run python scripts/reward_parity.py --preset original --seeds 3
    uv run --group warp python scripts/reward_parity.py --jax-backends jax warp
"""

import argparse
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

SB3_BACKENDS = ("dummy", "cpp", "ctypes", "envpool")


def run_sb3(backend, preset, obs_type, seed, timesteps, n_envs, n_steps, episodes):
    import torch
    from stable_baselines3.common.evaluation import evaluate_policy

    from elevator_rl.train import make_eval_env, make_model

    torch.set_num_threads(1)
    model = make_model(
        obs_type,
        reward_shaping=True,
        seed=seed,
        n_steps=n_steps,
        verbose=0,
        preset=preset,
        n_envs=n_envs,
        vec_env=backend,
        net_arch=[256, 256],
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    seconds = time.perf_counter() - t0
    eval_env = make_eval_env(obs_type, preset=preset)
    eval_env.seed(10_000 + seed)
    mean, _ = evaluate_policy(model, eval_env, n_eval_episodes=episodes)
    return backend, seed, mean, seconds


def run_jax(backend, preset, obs_type, seed, timesteps, n_envs, n_steps, episodes):
    from elevator_rl.jax_ppo import PPOConfig, train

    config = PPOConfig(n_envs=n_envs, n_steps=n_steps, net_arch=(256, 256))
    t0 = time.perf_counter()
    _, history = train(
        preset,
        timesteps,
        config,
        obs_type,
        seed=seed,
        eval_freq=timesteps,
        n_eval_episodes=episodes,
        verbose=False,
        backend=backend,
    )
    return f"jax_ppo/{backend}", seed, history[-1]["eval_mean"], time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="original")
    parser.add_argument("--obs-type", default="relative")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--backends", nargs="+", choices=SB3_BACKENDS, default=list(SB3_BACKENDS))
    parser.add_argument("--jax-backends", nargs="*", choices=("jax", "warp"), default=[])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    common = (args.preset, args.obs_type)
    rest = (args.timesteps, args.n_envs, args.n_steps, args.episodes)
    jobs = [(b, *common, s, *rest) for b in args.backends for s in range(args.seeds)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_sb3, *zip(*jobs, strict=True))) if jobs else []
    for b in args.jax_backends:
        results += [run_jax(b, *common, s, *rest) for s in range(args.seeds)]

    print(
        f"Preset {args.preset}, {args.obs_type} obs, {args.timesteps:,} steps, "
        f"{args.n_envs} envs x {args.n_steps} steps, {args.seeds} seeds, "
        f"{args.episodes} eval episodes on the Python env (jax_ppo: on its own env).\n"
    )
    print("| Backend | Reward per seed | Mean ± std | Minutes per run |")
    print("| --- | --- | --: | --: |")
    for backend in dict.fromkeys(r[0] for r in results):
        rows = [r for r in results if r[0] == backend]
        rewards = [r[2] for r in rows]
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        per_seed = ", ".join(f"{x:.1f}" for x in rewards)
        minutes = statistics.mean(r[3] for r in rows) / 60
        print(
            f"| {backend} | {per_seed} | {statistics.mean(rewards):.1f} ± {std:.1f} | "
            f"{minutes:.1f} |"
        )


if __name__ == "__main__":
    main()
