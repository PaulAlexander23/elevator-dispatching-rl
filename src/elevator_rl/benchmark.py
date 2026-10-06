"""Benchmark the observation spaces against each other and the baselines.

For each observation type, PPO is trained for a fixed number of timesteps
over several seeds and then evaluated on the unshaped environment, so every
reward is "passengers delivered per 200-step episode". Raw environment
throughput is measured too, since the observation mapping runs every step.

    uv run python -m elevator_rl.benchmark --timesteps 200000 --seeds 3 --readme README.md

With --readme, the results table replaces everything between the
BENCHMARK:START and BENCHMARK:END markers in that file.
"""

import argparse
import json
import os
import platform
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

from stable_baselines3.common.evaluation import evaluate_policy

from elevator_rl.baselines import RandomPolicy, UpDownPolicy
from elevator_rl.env import OBS_TYPES, LiftEnv
from elevator_rl.train import make_eval_env, make_model

START_MARKER = "<!-- BENCHMARK:START -->"
END_MARKER = "<!-- BENCHMARK:END -->"


def env_steps_per_second(obs_type, n_steps=20_000, seed=0):
    env = LiftEnv(obs_type=obs_type)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    return n_steps / (time.perf_counter() - t0)


def run_ppo(obs_type, seed, timesteps, n_eval_episodes, n_steps=2048):
    """Train and evaluate one PPO model. Runs in a worker process."""
    import torch

    torch.set_num_threads(1)
    model = make_model(obs_type, seed=seed, n_steps=n_steps, verbose=0)
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    train_seconds = time.perf_counter() - t0

    eval_env = make_eval_env(obs_type)
    eval_env.seed(10_000 + seed)
    rewards = {}
    for deterministic in (True, False):
        mean, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=deterministic
        )
        rewards[deterministic] = float(mean)
    return {
        "policy": f"PPO ({obs_type})",
        "obs_type": obs_type,
        "seed": seed,
        "reward_deterministic": rewards[True],
        "reward_stochastic": rewards[False],
        "train_steps_per_second": model.num_timesteps / train_seconds,
    }


def run_baseline(policy, n_eval_episodes, seed=0):
    eval_env = make_eval_env("box")
    eval_env.seed(seed)
    mean, _ = evaluate_policy(policy, eval_env, n_eval_episodes=n_eval_episodes)
    return {
        "policy": type(policy).__name__.removesuffix("Policy"),
        "obs_type": "box",
        "seed": seed,
        "reward_deterministic": float(mean),
        "reward_stochastic": float(mean),
        "train_steps_per_second": None,
    }


def summarise(runs, env_fps):
    """Group runs by policy: mean and standard deviation across seeds."""
    rows = []
    for policy in dict.fromkeys(run["policy"] for run in runs):
        group = [run for run in runs if run["policy"] == policy]
        row = {"policy": policy, "obs_type": group[0]["obs_type"], "seeds": len(group)}
        for key in ("reward_deterministic", "reward_stochastic", "train_steps_per_second"):
            values = [run[key] for run in group if run[key] is not None]
            row[key] = statistics.mean(values) if values else None
            row[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        trained = row["train_steps_per_second"] is not None
        row["env_steps_per_second"] = env_fps.get(row["obs_type"]) if trained else None
        rows.append(row)
    return rows


def _mean_std(row, key):
    if row[key] is None:
        return "–"
    if row["seeds"] > 1:
        return f"{row[key]:.1f} ± {row[key + '_std']:.1f}"
    return f"{row[key]:.1f}"


def _rate(value):
    return "–" if value is None else f"{value:,.0f}"


def markdown_table(rows, settings):
    lines = [
        f"_{settings['date']}: {settings['timesteps']:,} PPO timesteps × "
        f"{settings['seeds']} seed(s), {settings['episodes']} evaluation episodes, "
        f"{settings['machine']}._",
        "",
        "Reward is passengers delivered per 200-step episode (unshaped), "
        "as mean ± standard deviation across seeds.",
        "",
        "| Policy | Reward (deterministic) | Reward (stochastic) | Train steps/s | Env steps/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} "
            f"| {_mean_std(row, 'reward_deterministic')} "
            f"| {_mean_std(row, 'reward_stochastic')} "
            f"| {_rate(row['train_steps_per_second'])} "
            f"| {_rate(row['env_steps_per_second'])} |"
        )
    return "\n".join(lines)


def update_readme(path, table):
    path = Path(path)
    text = path.read_text()
    pattern = re.compile(re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{path} has no {START_MARKER} ... {END_MARKER} block")
    replacement = f"{START_MARKER}\n{table}\n{END_MARKER}"
    path.write_text(pattern.sub(lambda _: replacement, text, count=1))


def run_benchmark(obs_types, timesteps, seeds, episodes, workers, n_steps=2048):
    env_fps = {obs_type: env_steps_per_second(obs_type) for obs_type in obs_types}

    jobs = [(obs_type, seed) for obs_type in obs_types for seed in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_ppo, obs_type, seed, timesteps, episodes, n_steps)
            for obs_type, seed in jobs
        ]
        runs = []
        for (obs_type, seed), future in zip(jobs, futures, strict=True):
            runs.append(future.result())
            print(f"done: {obs_type} seed {seed}", flush=True)

    runs += [run_baseline(RandomPolicy(), episodes), run_baseline(UpDownPolicy(), episodes)]
    return runs, env_fps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--obs-types", nargs="+", choices=OBS_TYPES, default=list(OBS_TYPES))
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="parallel training processes"
    )
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout length")
    parser.add_argument("--readme", help="write the table into this file's benchmark block")
    parser.add_argument("--json", help="also save the raw per-seed results here")
    args = parser.parse_args(argv)

    runs, env_fps = run_benchmark(
        args.obs_types, args.timesteps, args.seeds, args.episodes, args.workers, args.n_steps
    )
    settings = {
        "date": date.today().isoformat(),
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    }
    table = markdown_table(summarise(runs, env_fps), settings)
    print(table)

    if args.json:
        Path(args.json).write_text(
            json.dumps({"settings": settings, "env_fps": env_fps, "runs": runs}, indent=2)
        )
    if args.readme:
        update_readme(args.readme, table)
        print(f"updated {args.readme}")


if __name__ == "__main__":
    main()
