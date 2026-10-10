import pytest

from elevator_rl import sweep


def test_sweep_smoke(capsys):
    runs = sweep.main(
        [
            "--preset", "multi",
            "--n-envs", "1", "4",
            "--rollout", "64",
            "--timesteps", "64",
            "--repeats", "2",
        ]
    )  # fmt: skip
    assert len(runs) == 4
    for run in runs:
        assert run["n_steps"] * run["n_envs"] == 64
        assert run["steps"] >= 64
        shares = run["env_share"] + run["policy_share"] + run["update_share"]
        assert shares == pytest.approx(1.0)
        assert 0 < run["env_share"] < 1
    assert "| dummy | 4 | 64 |" in capsys.readouterr().out


def test_rollout_must_cover_every_env():
    with pytest.raises(ValueError):
        sweep.run_one("original", n_envs=8, batch_size=4, rollout=4, timesteps=8)
