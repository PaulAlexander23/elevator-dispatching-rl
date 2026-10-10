"""Sweep PPO throughput over the number of parallel envs and the batch size.

Each run times where training spends its wall-clock time:
- env: stepping the vectorised env (measured inside training, by a wrapper);
- policy: the rest of rollout collection (forward passes, buffer, SB3 overhead);
- update: the gradient epochs after each rollout.

The rollout size (n_envs x n_steps) is held fixed, so only the parallelism
changes between runs. One rollout is run first as a warm-up and not timed.

    uv run python -m elevator_rl.sweep --preset full --n-envs 1 4 16 64
"""

import argparse
import json
import platform
import statistics
import time
from datetime import date
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper

from elevator_rl.building import PRESETS
from elevator_rl.building_env import OBS_TYPES
from elevator_rl.train import VEC_ENVS, make_vec_env


class TimedVecEnv(VecEnvWrapper):
    """Adds up the wall-clock time spent stepping and resetting the envs."""

    def __init__(self, venv):
        super().__init__(venv)
        self.seconds = 0.0

    def reset(self):
        t0 = time.perf_counter()
        obs = self.venv.reset()
        self.seconds += time.perf_counter() - t0
        return obs

    def step_async(self, actions):
        t0 = time.perf_counter()
        self.venv.step_async(actions)
        self.seconds += time.perf_counter() - t0

    def step_wait(self):
        t0 = time.perf_counter()
        result = self.venv.step_wait()
        self.seconds += time.perf_counter() - t0
        return result


class PhaseTimer(BaseCallback):
    """Adds up the time spent collecting rollouts (the rest is the update)."""

    def __init__(self):
        super().__init__()
        self.collect_seconds = 0.0
        self._start = None

    def _on_rollout_start(self):
        self._start = time.perf_counter()

    def _on_rollout_end(self):
        self.collect_seconds += time.perf_counter() - self._start

    def _on_step(self):
        return True


def run_one(
    preset,
    n_envs,
    batch_size,
    rollout,
    timesteps,
    vec_env="dummy",
    seed=0,
    obs_type="custom",
    net_arch=None,
):
    """Train once and return throughput and the time split. Runs one config."""
    n_steps = rollout // n_envs
    if n_steps < 1:
        raise ValueError(f"rollout {rollout} is smaller than n_envs {n_envs}")
    venv = make_vec_env(n_envs, vec_env, obs_type, True, preset, seed=seed)
    envs = TimedVecEnv(venv)
    model = PPO(
        "MlpPolicy",
        envs,
        device="cpu",
        verbose=0,
        n_epochs=3,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        policy_kwargs={"net_arch": net_arch} if net_arch else None,
    )
    model.learn(total_timesteps=n_steps * n_envs)  # warm-up, not timed

    envs.seconds = 0.0
    timer = PhaseTimer()
    start_steps = model.num_timesteps
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=timer, reset_num_timesteps=False)
    total = time.perf_counter() - t0
    steps = model.num_timesteps - start_steps
    envs.close()

    env = envs.seconds
    return {
        "preset": preset,
        "vec_env": vec_env,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "seed": seed,
        "steps": steps,
        "steps_per_second": steps / total,
        "env_share": env / total,
        "policy_share": (timer.collect_seconds - env) / total,
        "update_share": (total - timer.collect_seconds) / total,
    }


def summarise(runs):
    """Mean (and spread of the throughput) over seeds for each config."""
    rows = []
    keys = ("vec_env", "n_envs", "batch_size")
    for config in dict.fromkeys(tuple(run[k] for k in keys) for run in runs):
        group = [run for run in runs if tuple(run[k] for k in keys) == config]
        row = dict(zip(keys, config, strict=True))
        row["repeats"] = len(group)
        for key in ("steps_per_second", "env_share", "policy_share", "update_share"):
            row[key] = statistics.mean(run[key] for run in group)
        speeds = [run["steps_per_second"] for run in group]
        row["steps_per_second_std"] = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
        rows.append(row)
    return rows


def markdown_table(rows, settings):
    lines = [
        f"_{settings['date']}: preset `{settings['preset']}`, rollout {settings['rollout']:,} "
        f"steps, {settings['timesteps']:,} timed steps x {settings['repeats']} repeat(s), "
        f"{settings['torch_threads']} torch thread(s), {settings['machine']}._",
        "",
        "| VecEnv | n_envs | Batch | PPO steps/s | Env | Policy + SB3 | Update |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        speed = f"{row['steps_per_second']:,.0f}"
        if row["repeats"] > 1:
            speed += f" ± {row['steps_per_second_std']:,.0f}"
        lines.append(
            f"| {row['vec_env']} | {row['n_envs']} | {row['batch_size']} | {speed} "
            f"| {row['env_share']:.0%} | {row['policy_share']:.0%} | {row['update_share']:.0%} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=PRESETS, default="full")
    parser.add_argument("--n-envs", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--vec-envs", nargs="+", choices=VEC_ENVS, default=["dummy"])
    parser.add_argument("--obs-type", choices=OBS_TYPES, default="custom")
    parser.add_argument("--net-arch", type=int, nargs="+", help="hidden layer sizes")
    parser.add_argument("--rollout", type=int, default=2048, help="n_envs x n_steps")
    parser.add_argument("--timesteps", type=int, default=50_000, help="timed steps per run")
    parser.add_argument("--repeats", type=int, default=1, help="seeds per config")
    parser.add_argument(
        "--torch-threads", type=int, default=1, help="1 is usually fastest for small MLPs"
    )
    parser.add_argument("--json", help="also save the raw per-run results here")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(args.torch_threads)
    runs = []
    for vec_env in args.vec_envs:
        for batch_size in args.batch_sizes:
            for n_envs in args.n_envs:
                for seed in range(args.repeats):
                    run = run_one(
                        args.preset,
                        n_envs,
                        batch_size,
                        args.rollout,
                        args.timesteps,
                        vec_env,
                        seed,
                        args.obs_type,
                        args.net_arch,
                    )
                    runs.append(run)
                    print(
                        f"{vec_env} n_envs={n_envs} batch={batch_size} seed={seed}: "
                        f"{run['steps_per_second']:,.0f} steps/s, env {run['env_share']:.0%}",
                        flush=True,
                    )

    settings = {
        "date": date.today().isoformat(),
        "preset": args.preset,
        "obs_type": args.obs_type,
        "net_arch": args.net_arch,
        "rollout": args.rollout,
        "timesteps": args.timesteps,
        "repeats": args.repeats,
        "torch_threads": args.torch_threads,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    }
    table = markdown_table(summarise(runs), settings)
    print(table)
    if args.json:
        Path(args.json).write_text(json.dumps({"settings": settings, "runs": runs}, indent=2))
    return runs


if __name__ == "__main__":
    main()
