from collections import deque
from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from elevator_rl.building import (
    DOORS,
    DOWN,
    IDLE,
    MOVING,
    PRESETS,
    SERVE,
    SERVE_DOWN,
    SERVE_UP,
    UP,
    BuildingConfig,
    BuildingSim,
)
from elevator_rl.building_env import OBS_TYPES, BuildingEnv
from elevator_rl.env import LiftEnv


def quiet(config=None, **changes):
    """A config with no arrivals, so every passenger is placed by the test."""
    return replace(config or BuildingConfig(), arrival_probability=0.0, **changes)


def make_sim(config):
    return BuildingSim(config, rng=np.random.default_rng(0))


@pytest.mark.parametrize("obs_type", OBS_TYPES)
@pytest.mark.parametrize("reward_shaping", [False, True])
def test_default_config_matches_lift_env(obs_type, reward_shaping):
    original = LiftEnv(obs_type=obs_type, reward_shaping=reward_shaping)
    building = BuildingEnv(obs_type=obs_type, reward_shaping=reward_shaping)
    actions = np.random.default_rng(1).integers(3, size=450)

    obs_a, _ = original.reset(seed=7)
    obs_b, _ = building.reset(seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)
    for action in actions:
        obs_a, reward_a, _, truncated_a, _ = original.step(action)
        obs_b, reward_b, _, truncated_b, _ = building.step([action])
        np.testing.assert_array_equal(obs_a, obs_b)
        assert reward_a == reward_b
        assert truncated_a == truncated_b
        if truncated_a:
            original.reset()
            building.reset()


@pytest.mark.parametrize("obs_type", OBS_TYPES)
@pytest.mark.parametrize("preset", PRESETS)
def test_presets_pass_gymnasium_checks_and_stay_in_space(preset, obs_type):
    env = BuildingEnv(preset, obs_type=obs_type, reward_shaping=True)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(env.max_steps):
        assert env.observation_space.contains(obs), obs
        obs, _, _, _, info = env.step(env.action_space.sample())
    assert info["waiting"] >= 0


def test_presets_add_one_feature_each():
    names = list(PRESETS)
    for before, after in zip(names, names[1:], strict=False):
        a, b = PRESETS[before], PRESETS[after]
        changed = [f for f in a.__dataclass_fields__ if getattr(a, f) != getattr(b, f)]
        assert len(changed) == 1, (before, after, changed)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        BuildingConfig(traffic="rush")
    with pytest.raises(ValueError):
        BuildingConfig(n_lifts=0)
    with pytest.raises(ValueError):
        BuildingEnv(obs_type="multi_binary")


def test_lower_numbered_lift_boards_first():
    sim = make_sim(quiet(n_lifts=2, lift_capacity=2))
    sim.state().floor_queues[0].extend([3, 4, 5])
    delivered, boarded, served = sim.apply_actions([SERVE, SERVE])
    assert boarded == [2, 1] and served == [True, True]
    assert [lift.passengers for lift in sim.state().lifts] == [[3, 4], [5]]


def test_lifts_move_independently():
    sim = make_sim(quiet(n_lifts=3))
    sim.apply_actions([UP, UP, SERVE])
    sim.apply_actions([UP, DOWN, UP])
    assert [lift.floor for lift in sim.state().lifts] == [2, 0, 1]


def test_hall_calls_board_one_direction_in_arrival_order():
    sim = make_sim(quiet(hall_calls=True, lift_capacity=2))
    lift = sim.state().lifts[0]
    lift.position = lift.target = 5
    sim.state().floor_queues[5] = deque([2, 8, 1, 9, 7])

    _, boarded, _ = sim.apply_actions([SERVE_UP])
    assert boarded == [2] and lift.passengers == [8, 9]
    assert list(sim.state().floor_queues[5]) == [2, 1, 7]

    lift.passengers = []
    _, boarded, _ = sim.apply_actions([SERVE_DOWN])
    assert lift.passengers == [2, 1]
    assert list(sim.state().floor_queues[5]) == [7]


def test_hall_call_observation_splits_up_and_down():
    env = BuildingEnv(quiet(hall_calls=True), obs_type="custom")
    env.reset(seed=0)
    env.sim.state().floor_queues[3].extend([0, 5, 6, 1, 9])
    obs = env._observation()
    n = env.config.n_floors
    assert obs[2 + 3] == 3  # going up
    assert obs[2 + n + 3] == 2  # going down


def run_until_idle(sim, max_steps=100):
    lift = sim.state().lifts[0]
    for step in range(1, max_steps + 1):
        sim.apply_actions([SERVE])  # ignored while the lift is busy
        if lift.status == IDLE:
            return step
    raise AssertionError("lift never came to rest")


def test_one_floor_trip_takes_the_kinematic_time():
    config = quiet(kinematics=True)
    sim = make_sim(config)
    lift = sim.state().lifts[0]
    sim.apply_actions([UP])
    assert lift.status == MOVING and lift.target == 1
    # Triangular profile: 2 * sqrt(3.5 m / 1 m/s^2) = 3.74 s, so 4 one-second steps.
    steps = 1
    while lift.status != IDLE:
        sim.apply_actions([DOWN])  # a reversal mid-flight is ignored
        steps += 1
    assert steps == 4
    assert lift.position == 1.0 and lift.velocity == 0.0


def test_moving_lift_can_extend_its_trip_and_respects_max_speed():
    config = quiet(kinematics=True)
    sim = make_sim(config)
    lift = sim.state().lifts[0]
    max_speed = config.max_speed / config.floor_height
    top_speed = 0.0
    for _ in range(12):
        sim.apply_actions([UP])
        assert lift.target - lift.position <= 2
        top_speed = max(top_speed, abs(lift.velocity))
    assert lift.target > 3
    assert top_speed == pytest.approx(max_speed)


def test_doors_block_the_lift_for_door_time():
    config = quiet(kinematics=True)
    sim = make_sim(config)
    lift = sim.state().lifts[0]
    lift.passengers = [0, 4]
    delivered, _, served = sim.apply_actions([SERVE])
    assert delivered == [1] and served == [True]
    assert lift.status == DOORS
    steps = 1
    while lift.status != IDLE:
        _, _, served = sim.apply_actions([UP])
        assert served == [False]
        steps += 1
    assert steps == int(config.door_time / config.step_seconds)
    assert lift.floor == 0
    sim.apply_actions([UP])
    assert lift.status == MOVING


def test_kinematic_shaping_ignores_busy_lifts():
    env = BuildingEnv(quiet(kinematics=True), reward_shaping=True)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step([SERVE])
    assert reward == -0.5
    _, reward, _, _, _ = env.step([SERVE])  # doors still open: no new serve
    assert reward == 0


@pytest.mark.parametrize("traffic", ["up_peak", "down_peak", "lunch", "day"])
def test_arrival_model_is_normalised(traffic):
    config = BuildingConfig(n_floors=12, traffic=traffic)
    sim = make_sim(config)
    for fraction in (0.0, 0.4, 0.8, 1.0):
        rates, destinations = sim.arrival_model(fraction)
        assert rates.sum() == pytest.approx(config.arrival_probability * config.n_floors)
        np.testing.assert_allclose(destinations.sum(axis=1), 1.0)
        assert np.all(np.diag(destinations)[rates > 0] == 0)


def test_up_peak_starts_at_the_lobby_and_day_turns_to_down_peak():
    sim = make_sim(BuildingConfig(traffic="day"))
    morning, _ = sim.arrival_model(0.0)
    evening, destinations = sim.arrival_model(1.0)
    assert morning[0] > morning[1:].sum()
    assert evening[0] < evening[1:].sum()
    assert destinations[5, 0] > 0.5


def test_profile_arrivals_are_valid():
    config = BuildingConfig(n_floors=15, traffic="day")
    sim = make_sim(config)
    for step in range(200):
        sim.sample_passengers(step / 200)
    total = 0
    for floor, queue in enumerate(sim.state().floor_queues):
        total += len(queue)
        assert all(0 <= d < config.n_floors and d != floor for d in queue)
    # About 1.5 arrivals per step on average.
    assert 200 < total < 400


def test_human_render_prints_every_floor(capsys):
    env = BuildingEnv("full", render_mode="human")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    env.render()
    assert capsys.readouterr().out.count("|") == env.config.n_floors
