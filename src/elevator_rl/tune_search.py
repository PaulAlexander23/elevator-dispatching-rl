"""Optuna search over the JAX PPO config, with pruning on the eval curve.

    uv run --group warp --group hydra python -m elevator_rl.tune_search \\
        --trials 50 --name full-lr preset=full timesteps=5000000

Trailing arguments are Hydra overrides of `conf/tune.yaml`, applied to every
trial; the search space is `conf/search.yaml` (or `--space`). Each trial
writes `result.json` under `runs/tune/search/<name>/<trial>`, so
`elevator_rl.tune_report` reads a search as it does a grid sweep. The study is
kept in SQLite beside them, so a search can be stopped and resumed.

Hydra's own Optuna sweeper plugin pins optuna<3, so this drives Optuna 5
directly through Hydra's compose API instead.
"""

import argparse
import json
from pathlib import Path

import optuna
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from elevator_rl.tune import run

DEFAULT_SPACE = Path(__file__).parent / "conf" / "search.yaml"


def suggest(trial, space):
    """Overrides for one trial, as Hydra override strings."""
    overrides = []
    for key, spec in space.items():
        if "choices" in spec:
            # Optuna's choices must be scalars, so lists are picked by index.
            choices = [json.dumps(c, separators=(",", ":")) for c in spec["choices"]]
            value = trial.suggest_categorical(key, choices)
        elif isinstance(spec["low"], int) and isinstance(spec["high"], int):
            value = trial.suggest_int(key, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            value = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
            value = float(f"{value:.3g}")  # readable overrides; finer is noise
        overrides.append(f"{key}={value}")
    return overrides


def make_objective(base_overrides, space, out_dir):
    def objective(trial):
        overrides = base_overrides + suggest(trial, space)
        with initialize_config_module("elevator_rl.conf", version_base=None):
            cfg = compose("tune", overrides=overrides)
        print(f"trial {trial.number}: {' '.join(overrides)}", flush=True)

        def report(row):
            trial.report(row["eval_mean"], row["timesteps"])
            return trial.should_prune()

        try:
            result = run(cfg, callback=report, verbose=False)
        except ValueError as e:  # e.g. a minibatch that does not divide the rollout
            raise optuna.TrialPruned(str(e)) from e
        result["overrides"] = overrides
        result["pruned"] = bool(trial.should_prune())
        trial_dir = out_dir / f"{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(json.dumps(result, indent=1))
        print(f"  score {result['score']:.1f}{' (pruned)' if result['pruned'] else ''}", flush=True)
        if result["pruned"]:
            raise optuna.TrialPruned()
        return result["score"]

    return objective


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("overrides", nargs="*", help="Hydra overrides for every trial")
    parser.add_argument("--name", default="search", help="study name and output folder")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument("--seed", type=int, default=0, help="sampler seed")
    parser.add_argument("--startup", type=int, default=5, help="trials before pruning starts")
    parser.add_argument(
        "--warmup", type=int, default=1_000_000, help="steps before a trial can be pruned"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/tune/search"))
    args = parser.parse_args(argv)

    out_dir = args.out / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    space = OmegaConf.to_container(OmegaConf.load(args.space))
    study = optuna.create_study(
        study_name=args.name,
        storage=f"sqlite:///{out_dir / 'study.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.startup, n_warmup_steps=args.warmup
        ),
    )
    study.optimize(make_objective(args.overrides, space, out_dir), n_trials=args.trials)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        print(f"best score {study.best_value:.1f}: {study.best_params}")
    return study


if __name__ == "__main__":
    main()
