import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from elevator_rl.building_env import BuildingEnv
from elevator_rl.imitate import collect, pretrain


def test_collect_records_the_heuristic():
    obs, actions, returns = collect("multi", 300, obs_type="relative")
    assert obs.shape[0] == actions.shape[0] == returns.shape[0] == 300
    assert actions.shape[1] == 4
    assert obs.dtype == np.float32
    assert np.isfinite(returns).all()


def test_pretraining_raises_the_likelihood_of_the_heuristic():
    obs, actions, returns = collect("original", 2000, obs_type="relative")
    model = PPO(
        "MlpPolicy",
        DummyVecEnv([lambda: BuildingEnv("original", obs_type="relative")]),
        device="cpu",
        seed=0,
    )
    first, _ = pretrain(model, obs, actions, returns, epochs=1)
    last, _ = pretrain(model, obs, actions, returns, epochs=5)
    assert last > first
