"""Hyperparameter testing for the JAX PPO, configured with Hydra.

One run trains `jax_ppo` with the settings in `conf/tune.yaml` and writes
`result.json` (the config, the eval history and the final score) to its
output directory. Hydra's multirun turns that into a grid sweep:

    uv run --group warp --group hydra python -m elevator_rl.tune -m \\
        ppo.learning_rate=1e-4,3e-4,1e-3 seed=0,1,2

`python -m elevator_rl.tune_report <multirun dir>` then tabulates the sweep,
averaging over seeds. For a search rather than a grid, `elevator_rl.tune_search`
runs Optuna over the same config.
"""

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from elevator_rl.jax_ppo import PPOConfig, train


def final_score(history, tail=0.2):
    """Mean eval reward over the last `tail` of the evals, as the plan doc reports."""
    n = max(round(len(history) * tail), 1)
    return sum(row["eval_mean"] for row in history[-n:]) / n


def ppo_config(cfg: DictConfig) -> PPOConfig:
    ppo = OmegaConf.to_container(cfg.ppo)
    ppo["net_arch"] = tuple(ppo["net_arch"])
    return PPOConfig(**ppo)


def warm_start_params(cfg: DictConfig, config: PPOConfig):
    """Params cloned from the heuristic, or None when `warm_start.samples` is 0."""
    if not cfg.warm_start.samples:
        return None, {}
    from elevator_rl.jax_env import JaxBuildingEnv
    from elevator_rl.jax_imitate import warm_start

    env = JaxBuildingEnv(cfg.preset, obs_type=cfg.obs_type)  # only its sizes are used
    return warm_start(
        env,
        cfg.preset,
        config,
        cfg.warm_start.samples,
        epochs=cfg.warm_start.epochs,
        seed=cfg.seed,
        cache_dir=cfg.warm_start.cache_dir,
    )


def run(cfg: DictConfig, callback=None, verbose=True):
    """Train with `cfg` and return a result dict (config, history, score)."""
    config = ppo_config(cfg)
    params, imitation = warm_start_params(cfg, config)
    _, history = train(
        cfg.preset,
        cfg.timesteps,
        config,
        cfg.obs_type,
        reward_shaping=cfg.reward_shaping,
        seed=cfg.seed,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        verbose=verbose,
        backend=cfg.backend,
        callback=callback,
        params=params,
    )
    return {
        "imitation": imitation,
        "config": OmegaConf.to_container(cfg),
        "history": history,
        "score": final_score(history),
        "best": max(row["eval_mean"] for row in history),
        "seconds": history[-1]["seconds"],
        "steps_per_second": history[-1]["steps_per_second"],
    }


@hydra.main(config_path="conf", config_name="tune", version_base=None)
def main(cfg: DictConfig) -> float:
    result = run(cfg)
    hc = HydraConfig.get()
    result["overrides"] = list(hc.overrides.task)
    out = Path(hc.runtime.output_dir) / "result.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"score {result['score']:.1f} -> {out}")
    return result["score"]


if __name__ == "__main__":
    main()
