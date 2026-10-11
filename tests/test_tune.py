"""Hyperparameter testing (elevator_rl.tune*). Skipped unless JAX and Hydra are installed."""

import json

import pytest

pytest.importorskip("jax")
pytest.importorskip("hydra")
optuna = pytest.importorskip("optuna")

from hydra import compose, initialize_config_module  # noqa: E402

from elevator_rl import tune, tune_report, tune_search  # noqa: E402

TINY = [
    "preset=multi",
    "backend=jax",
    "timesteps=1024",
    "eval_freq=512",
    "n_eval_episodes=2",
    "ppo.n_envs=8",
    "ppo.n_steps=16",
    "ppo.batch_size=64",
    "ppo.net_arch=[16,16]",
]


def compose_tune(overrides):
    with initialize_config_module("elevator_rl.conf", version_base=None):
        return compose("tune", overrides=overrides)


def test_config_builds_the_ppo_config():
    cfg = compose_tune(["ppo.learning_rate=1e-4", "ppo.net_arch=[32,32]"])
    config = tune.ppo_config(cfg)
    assert config.learning_rate == 1e-4
    assert config.net_arch == (32, 32)
    assert cfg.timesteps == 5_000_000


def test_final_score_is_the_mean_of_the_last_fifth():
    history = [{"eval_mean": float(x)} for x in range(10)]
    assert tune.final_score(history) == pytest.approx(8.5)
    assert tune.final_score(history[:1]) == 0.0


def test_run_returns_a_result():
    result = tune.run(compose_tune(TINY), verbose=False)
    assert len(result["history"]) == 2
    assert result["config"]["ppo"]["n_envs"] == 8
    json.dumps(result)


def test_report_groups_seeds():
    def result(lr, seed, score):
        return {
            "overrides": ["preset=multi", f"ppo.learning_rate={lr}", f"seed={seed}"],
            "score": score,
            "best": score,
            "seconds": 60.0,
            "steps_per_second": 1000.0,
        }

    results = [result(1e-4, 0, 10.0), result(1e-4, 1, 20.0), result(3e-4, 0, 30.0)]
    assert tune_report.common_overrides(results) == {"preset=multi"}
    rows = tune_report.summarise(results)
    assert [r["setting"] for r in rows] == ["ppo.learning_rate=0.0003", "ppo.learning_rate=0.0001"]
    assert rows[1]["seeds"] == 2 and rows[1]["score"] == 15.0
    assert "| ppo.learning_rate=0.0001 | 2 | 15.0 ± 7.1 |" in tune_report.table(rows)


def test_search_writes_results(tmp_path):
    space = tmp_path / "space.yaml"
    space.write_text(
        "ppo.learning_rate: {low: 1.0e-4, high: 1.0e-3, log: true}\n"
        "ppo.net_arch: {choices: [[8, 8], [16, 16]]}\n"
    )
    study = tune_search.main(
        ["--trials", "2", "--name", "t", "--out", str(tmp_path), "--space", str(space), *TINY]
    )
    assert len(study.trials) == 2
    results = tune_report.load([tmp_path / "t"])
    assert len(results) == 2
    assert any(o.startswith("ppo.net_arch=[") for o in results[0]["overrides"])


def test_report_importance(tmp_path, capsys):
    tune_search.main(["--trials", "3", "--name", "imp", "--out", str(tmp_path), *TINY])
    tune_report.main([str(tmp_path / "imp"), "--importance"])
    out = capsys.readouterr().out
    assert "| Hyperparameter | Importance |" in out
