import pytest

from elevator_rl.train import main


def test_main_smoke(tmp_path):
    """Train for a single tiny rollout to check the pipeline runs end to end."""
    save_path = tmp_path / "model.zip"
    main(
        total_timesteps=64,
        eval_freq=64,
        n_eval_episodes=1,
        save_path=save_path,
        n_steps=64,
    )
    assert save_path.exists()


@pytest.mark.slow
def test_main_full(tmp_path):
    """The full training run (~30 minutes). Run with `pytest -m slow`."""
    main(save_path=tmp_path / "model.zip")
