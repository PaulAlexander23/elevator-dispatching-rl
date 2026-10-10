import pytest

from elevator_rl import compare


def test_parse_setting():
    assert compare.parse_setting("64x32:512") == {"n_envs": 64, "n_steps": 32, "batch_size": 512}
    with pytest.raises(ValueError):
        compare.parse_setting("64:512")


def test_compare_smoke(capsys):
    runs = compare.main(
        [
            "--preset", "multi",
            "--settings", "1x64:32", "4x16:32",
            "--timesteps", "128",
            "--seeds", "1",
            "--eval-freq", "64",
            "--episodes", "1",
            "--workers", "2",
        ]
    )  # fmt: skip
    assert len(runs) == 2
    for run in runs:
        curve = run["curve"]
        assert curve[0]["timesteps"] == 0
        assert curve[-1]["timesteps"] >= 128
        assert all(a["seconds"] <= b["seconds"] for a, b in zip(curve, curve[1:], strict=False))
    assert "| 4x16:32 |" in capsys.readouterr().out
