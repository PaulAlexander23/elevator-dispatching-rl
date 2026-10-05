import numpy as np
import pytest
from gymnasium import Env
from gymnasium.utils.env_checker import check_env

from elevator_rl.env import OBS_TYPES, LiftEnv
from elevator_rl.sim import Passenger

UP, DOWN, SERVE = 0, 1, 2


@pytest.mark.parametrize("obs_type", OBS_TYPES)
def test_env_passes_gymnasium_checks(obs_type):
    env = LiftEnv(obs_type=obs_type)
    assert isinstance(env, Env)
    check_env(env, skip_render_check=True)


def test_invalid_obs_type_is_rejected():
    with pytest.raises(ValueError):
        LiftEnv(obs_type="nope")


@pytest.mark.parametrize("obs_type", OBS_TYPES)
def test_observations_stay_in_space(obs_type):
    env = LiftEnv(obs_type=obs_type, reward_shaping=True)
    obs, _ = env.reset(seed=0)
    env.action_space.seed(0)
    assert env.observation_space.contains(obs)
    for _ in range(env.max_steps):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), obs


def test_episode_truncates_at_max_steps():
    env = LiftEnv(max_steps=20)
    env.reset(seed=0)
    for step in range(1, 21):
        _, _, terminated, truncated, _ = env.step(UP)
        assert not terminated
        assert truncated == (step == 20)


def test_reset_with_seed_is_reproducible():
    env = LiftEnv(obs_type="custom")
    trajectories = []
    for _ in range(2):
        obs, _ = env.reset(seed=123)
        observations = [obs]
        for action in [UP, SERVE, UP, UP, SERVE, DOWN, SERVE] * 5:
            obs, _, _, _, _ = env.step(action)
            observations.append(obs)
        trajectories.append(np.array(observations))
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


def empty_env(**kwargs):
    """An env with no random arrivals, so rewards are deterministic."""
    env = LiftEnv(**kwargs)
    env.reset(seed=0)
    env.sim.reset()
    env.sim.floor_probabilities = [0.0] * env.sim.n_floors
    return env


def test_reward_counts_passengers_delivered():
    env = empty_env()
    env.sim.state().lift_passengers = [Passenger(0, 1), Passenger(0, 1)]
    env.step(UP)
    _, reward, _, _, _ = env.step(SERVE)
    assert reward == 2


def test_reward_shaping_rewards_pickups_and_penalises_empty_serves():
    env = empty_env(reward_shaping=True)
    _, reward, _, _, _ = env.step(SERVE)
    assert reward == -0.5

    env.sim.state().floor_passengers[0].append(Passenger(0, 4))
    _, reward, _, _, _ = env.step(SERVE)
    assert reward == 1

    _, reward, _, _, _ = env.step(UP)
    assert reward == 0


def test_observation_contents():
    env = empty_env(obs_type="custom")
    state = env.sim.state()
    state.lift_position = 2
    state.lift_passengers = [Passenger(0, 5), Passenger(1, 5), Passenger(0, 7)]
    state.floor_passengers[4].extend([Passenger(4, 0)] * 3)

    obs = env._map_state_to_obs(state)
    n = env.sim.n_floors
    assert obs[0] == 2
    assert obs[1] == 3
    assert obs[2 + 4] == 3
    assert obs[2 + n + 5] == 2
    assert obs[2 + n + 7] == 1


def test_multi_discrete_shows_empty_lift_position():
    env = empty_env(obs_type="multi_discrete")
    env.sim.state().lift_position = 6
    obs = env._map_state_to_obs(env.sim.state())
    assert obs[6] == 1
    assert obs[: env.sim.n_floors].sum() == 1


def test_human_render_prints_state(capsys):
    env = LiftEnv(render_mode="human")
    env.reset(seed=0)
    env.step(SERVE)
    env.render()
    assert "|" in capsys.readouterr().out
