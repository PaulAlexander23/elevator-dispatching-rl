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


def test_main_smoke_with_preset(tmp_path):
    """The multi-lift env trains through the same pipeline."""
    save_path = tmp_path / "model.zip"
    main(
        total_timesteps=64,
        eval_freq=64,
        n_eval_episodes=1,
        save_path=save_path,
        n_steps=64,
        preset="full",
    )
    assert save_path.exists()


def test_main_smoke_with_imitation(tmp_path):
    """The recipe that learns the full preset: relative obs, imitation, low learning rate."""
    save_path = tmp_path / "model.zip"
    main(
        total_timesteps=128,
        eval_freq=128,
        n_eval_episodes=1,
        save_path=save_path,
        n_steps=16,
        n_envs=4,
        obs_type="relative",
        preset="full",
        net_arch=[32, 32],
        learning_rate=3e-5,
        pretrain=400,
    )
    assert save_path.exists()


def test_torch_threads(tmp_path, capsys):
    import torch

    before = torch.get_num_threads()
    try:
        main(total_timesteps=64, eval_freq=64, n_eval_episodes=1, save_path=tmp_path / "m.zip",
             n_steps=64, torch_threads=2)  # fmt: skip
        assert torch.get_num_threads() == 2
        assert "torch threads: 2" in capsys.readouterr().out
    finally:
        torch.set_num_threads(before)
