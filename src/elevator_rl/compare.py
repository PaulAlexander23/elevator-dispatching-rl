"""Compare how well PPO learns under different parallelism settings.

`elevator_rl.sweep` measures throughput only. This trains each setting for the
same number of env steps and evaluates the policy on the unshaped env as it
goes, so the reward can be compared per env step and per second of training.

A setting is written N_ENVSxN_STEPS:BATCH, for example 64x32:512 = 64 envs,
32 steps per env per rollout, minibatches of 512.

    uv run python -m elevator_rl.compare --preset full --timesteps 500000 \\
        --settings 1x2048:64 64x32:64 64x32:512 64x128:512 --seeds 3
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

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from elevator_rl.building import PRESETS
from elevator_rl.train import make_env, make_eval_env

SETTING = re.compile(r"^(\d+)x(\d+):(\d+)$")


def parse_setting(text):
    match = SETTING.match(text)
    if not match:
        raise ValueError(f"setting must look like 64x32:512, got {text!r}")
    n_envs, n_steps, batch_size = map(int, match.groups())
    return {"n_envs": n_envs, "n_steps": n_steps, "batch_size": batch_size}


class PeriodicEval(BaseCallback):
    """Evaluate every `eval_freq` env steps; the eval time is not counted."""

    def __init__(self, eval_env, eval_freq, n_episodes):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.curve = []
        self._next = 0
        self._eval_seconds = 0.0
        self._start = None

    def _on_training_start(self):
        self._start = time.perf_counter()
        self._evaluate()

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._evaluate()
        return True

    def _on_training_end(self):
        if not self.curve or self.curve[-1]["timesteps"] != self.num_timesteps:
            self._evaluate()

    def _evaluate(self):
        t0 = time.perf_counter()
        self.eval_env.seed(12_345)
        rewards, _ = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=self.n_episodes, return_episode_rewards=True
        )
        self._eval_seconds += time.perf_counter() - t0
        self.curve.append(
            {
                "timesteps": self.num_timesteps,
                "seconds": time.perf_counter() - self._start - self._eval_seconds,
                "reward": float(np.mean(rewards)),
            }
        )
        self._next = self.num_timesteps + self.eval_freq


def run_setting(preset, setting, seed, timesteps, eval_freq, n_eval_episodes):
    """Train one setting with one seed. Runs in a worker process."""
    import torch

    torch.set_num_threads(1)
    envs = DummyVecEnv(
        [lambda: make_env("custom", reward_shaping=True, preset=preset)] * setting["n_envs"]
    )
    model = PPO(
        "MlpPolicy",
        envs,
        device="cpu",
        verbose=0,
        n_epochs=3,
        n_steps=setting["n_steps"],
        batch_size=setting["batch_size"],
        seed=seed,
    )
    callback = PeriodicEval(make_eval_env("custom", preset=preset), eval_freq, n_eval_episodes)
    model.learn(total_timesteps=timesteps, callback=callback)
    return {**setting, "preset": preset, "seed": seed, "curve": callback.curve}


def random_baseline(preset, n_episodes, seed=12_345):
    env = make_env("custom", preset=preset)
    env.action_space.seed(seed)
    totals = []
    for episode in range(n_episodes):
        env.reset(seed=seed + episode)
        total, truncated = 0.0, False
        while not truncated:
            _, reward, _, truncated, _ = env.step(env.action_space.sample())
            total += reward
        totals.append(total)
    return float(np.mean(totals))


def label(run):
    return f"{run['n_envs']}x{run['n_steps']}:{run['batch_size']}"


def summarise(runs, final_fraction=0.2):
    """Per setting: final reward (mean of the last 20% of evals) and throughput."""
    rows = []
    for name in dict.fromkeys(label(run) for run in runs):
        group = [run for run in runs if label(run) == name]
        finals, speeds = [], []
        for run in group:
            curve = run["curve"]
            tail = curve[-max(1, round(len(curve) * final_fraction)) :]
            finals.append(statistics.mean(point["reward"] for point in tail))
            speeds.append(curve[-1]["timesteps"] / curve[-1]["seconds"])
        rows.append(
            {
                "setting": name,
                "seeds": len(group),
                "final_reward": statistics.mean(finals),
                "final_reward_std": statistics.stdev(finals) if len(finals) > 1 else 0.0,
                "steps_per_second": statistics.mean(speeds),
                "minutes": statistics.mean(run["curve"][-1]["seconds"] for run in group) / 60,
            }
        )
    return rows


def markdown_table(rows, settings):
    lines = [
        f"_{settings['date']}: preset `{settings['preset']}`, {settings['timesteps']:,} env steps "
        f"x {settings['seeds']} seed(s), {settings['episodes']} eval episodes every "
        f"{settings['eval_freq']:,} steps, {settings['machine']}. "
        f"Random policy: {settings['random_reward']:.1f}._",
        "",
        "| Setting (envs x steps : batch) | Final reward | PPO steps/s | Training minutes |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        reward = f"{row['final_reward']:.1f}"
        if row["seeds"] > 1:
            reward += f" ± {row['final_reward_std']:.1f}"
        lines.append(
            f"| {row['setting']} | {reward} | {row['steps_per_second']:,.0f} "
            f"| {row['minutes']:.1f} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=PRESETS, default="full")
    parser.add_argument(
        "--settings", nargs="+", default=["1x2048:64", "64x32:64", "64x32:512", "64x128:512"]
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--episodes", type=int, default=10, help="episodes per evaluation")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--json", help="also save the learning curves here")
    args = parser.parse_args(argv)

    settings_list = [parse_setting(text) for text in args.settings]
    jobs = [(s, seed) for s in settings_list for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_setting,
                args.preset,
                setting,
                seed,
                args.timesteps,
                args.eval_freq,
                args.episodes,
            )
            for setting, seed in jobs
        ]
        runs = []
        for future in futures:
            runs.append(future.result())
            print(f"done: {label(runs[-1])} seed {runs[-1]['seed']}", flush=True)

    settings = {
        "date": date.today().isoformat(),
        "preset": args.preset,
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "eval_freq": args.eval_freq,
        "random_reward": random_baseline(args.preset, args.episodes * 5),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    }
    table = markdown_table(summarise(runs), settings)
    print(table)
    if args.json:
        Path(args.json).write_text(json.dumps({"settings": settings, "runs": runs}, indent=2))
    return runs


if __name__ == "__main__":
    main()
