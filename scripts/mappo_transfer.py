"""MAPPO across lift counts: train with 4 lifts, evaluate with 2, 3, 6 and 8.

Trains a MAPPO policy on `full` with the tuned settings (conf/tuned.yaml) and
the imitation warm start, then moves its shared actor (`jax_ppo.transfer`)
into buildings with other lift counts and evaluates it there with no further
training, next to the collective heuristic in the same building. With
`--finetune N` it also trains the moved policy for N steps in each building
and, for comparison, a policy warm-started from scratch there for N steps.

    uv run --group warp python scripts/mappo_transfer.py --timesteps 20000000 \\
        --finetune 5000000 --seeds 0 1 2

Writes runs/mappo_transfer/<lifts>-<seed>.json and prints a markdown table.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from elevator_rl.baselines import CollectivePolicy
from elevator_rl.building import PRESETS
from elevator_rl.building_env import BuildingEnv
from elevator_rl.jax_env import JaxBuildingEnv
from elevator_rl.jax_imitate import warm_start
from elevator_rl.jax_ppo import PPOConfig, make_evaluate, train, transfer

TUNED = PPOConfig(
    n_envs=1024,
    n_steps=8,
    batch_size=2048,
    n_epochs=10,
    learning_rate=5.21e-5,
    clip_range=0.1,
    ent_coef=2.77e-5,
    vf_coef=0.25,
    net_arch=(256, 256),
    algo="mappo",
)


def heuristic(config, episodes):
    env = Monitor(BuildingEnv(config, obs_type="custom"))
    rewards, _ = evaluate_policy(
        CollectivePolicy(config), env, n_eval_episodes=episodes, return_episode_rewards=True
    )
    return float(np.mean(rewards))


def evaluate(params, config, episodes, seed):
    env = JaxBuildingEnv(config)  # unshaped: the return is passengers delivered
    rewards = jax.jit(make_evaluate(env, episodes))(params, jax.random.key(1000 + seed))
    return float(np.mean(rewards))


def tail(history, fraction=0.2):
    n = max(round(len(history) * fraction), 1)
    return float(np.mean([row["eval_mean"] for row in history[-n:]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--lifts", type=int, nargs="+", default=[2, 3, 6, 8])
    parser.add_argument("--finetune", type=int, default=0, help="steps of training per building")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--pretrain", type=int, default=100_000, help="warm-start samples")
    parser.add_argument("--no-lift-id", action="store_true")
    parser.add_argument("--backend", choices=("jax", "warp"), default="warp")
    parser.add_argument("--out", default="runs/mappo_transfer")
    args = parser.parse_args()
    config = TUNED._replace(lift_id=not args.no_lift_id)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        env = JaxBuildingEnv("full")
        params, _ = warm_start(
            env, "full", config, args.pretrain, seed=seed, cache_dir="runs/cache"
        )
        params, history = train(
            "full", args.timesteps, config, seed=seed, backend=args.backend,
            eval_freq=1_000_000, params=params, verbose=False,
        )  # fmt: skip
        print(f"seed {seed}: 4 lifts, eval {tail(history):.1f}", flush=True)
        for n_lifts in args.lifts:
            building = replace(PRESETS["full"], n_lifts=n_lifts)
            target = JaxBuildingEnv(building)
            moved = transfer(params, target, config, jax.random.key(seed))
            row = {
                "seed": seed,
                "lifts": n_lifts,
                "heuristic": heuristic(building, args.episodes),
                "zero_shot": evaluate(moved, building, args.episodes, seed),
            }
            if args.finetune:
                _, tuned = train(
                    building, args.finetune, config, seed=seed, backend=args.backend,
                    eval_freq=500_000, params=moved, verbose=False,
                )  # fmt: skip
                fresh, _ = warm_start(target, building, config, args.pretrain, seed=seed)
                _, scratch = train(
                    building, args.finetune, config, seed=seed, backend=args.backend,
                    eval_freq=500_000, params=fresh, verbose=False,
                )  # fmt: skip
                row |= {"finetuned": tail(tuned), "scratch": tail(scratch)}
            rows.append(row)
            (out / f"{n_lifts}-{seed}.json").write_text(json.dumps(row, indent=1))
            print(row, flush=True)

    columns = ["heuristic", "zero_shot"] + (["finetuned", "scratch"] if args.finetune else [])
    print("\n| Lifts | " + " | ".join(columns) + " |")
    print("| --: |" + " --: |" * len(columns))
    for n_lifts in args.lifts:
        cells = []
        for c in columns:
            values = [r[c] for r in rows if r["lifts"] == n_lifts]
            cells.append(f"{np.mean(values):.1f} ± {np.std(values):.1f}")
        print(f"| {n_lifts} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
