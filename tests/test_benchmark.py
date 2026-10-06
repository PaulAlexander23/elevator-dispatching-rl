import pytest

from elevator_rl import benchmark


def test_benchmark_smoke_updates_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Title\n\n{benchmark.START_MARKER}\nold\n{benchmark.END_MARKER}\n\nAfter.\n"
    )
    benchmark.main(
        [
            "--obs-types", "custom", "box",
            "--timesteps", "64",
            "--n-steps", "64",
            "--seeds", "2",
            "--episodes", "1",
            "--workers", "2",
            "--readme", str(readme),
        ]
    )  # fmt: skip
    text = readme.read_text()
    assert "old" not in text
    assert text.startswith("# Title") and text.endswith("After.\n")
    for policy in ("PPO (custom)", "PPO (box)", "Random", "UpDown"):
        assert f"| {policy} |" in text


def test_update_readme_requires_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here")
    with pytest.raises(ValueError):
        benchmark.update_readme(readme, "table")
