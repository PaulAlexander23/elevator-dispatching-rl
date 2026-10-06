from collections import deque

import numpy as np

from elevator_rl.sim import LiftSim, LiftState, Passenger


def make_sim(seed=0, **kwargs):
    return LiftSim(rng=np.random.default_rng(seed), **kwargs)


def test_state_starts_empty_at_ground_floor():
    state = LiftState(10, 8)
    assert state.lift_position == 0
    assert state.lift_passengers == []
    assert len(state.floor_passengers) == 10
    assert all(len(queue) == 0 for queue in state.floor_passengers)


def test_sample_passengers_are_valid():
    sim = make_sim()
    for _ in range(100):
        sim.sample_passengers()
    passengers = [p for queue in sim.state().floor_passengers for p in queue]
    assert passengers, "expected some arrivals in 100 steps at p=0.1 per floor"
    for floor, queue in enumerate(sim.state().floor_passengers):
        for passenger in queue:
            assert passenger.origin == floor
            assert 0 <= passenger.destination < sim.n_floors
            assert passenger.destination != passenger.origin


def test_sampling_is_reproducible_with_seed():
    a, b = make_sim(seed=42), make_sim(seed=42)
    for _ in range(50):
        a.sample_passengers()
        b.sample_passengers()
    assert a.state().floor_passengers == b.state().floor_passengers


def test_move_up_stops_at_top_floor():
    sim = make_sim()
    sim.move_up()
    assert sim.state().lift_position == 1
    sim.state().lift_position = sim.n_floors - 1
    sim.move_up()
    assert sim.state().lift_position == sim.n_floors - 1


def test_move_down_stops_at_ground_floor():
    sim = make_sim()
    sim.state().lift_position = 5
    sim.move_down()
    assert sim.state().lift_position == 4
    sim.state().lift_position = 0
    sim.move_down()
    assert sim.state().lift_position == 0


def test_serve_floor_drops_off_and_picks_up():
    sim = make_sim()
    state = sim.state()
    state.lift_position = 3
    state.lift_passengers = [Passenger(0, 3), Passenger(0, 5)]
    state.floor_passengers[3] = deque([Passenger(3, 7)])

    served, boarded = sim.serve_floor()

    assert (served, boarded) == (1, 1)
    assert state.lift_passengers == [Passenger(0, 5), Passenger(3, 7)]
    assert len(state.floor_passengers[3]) == 0


def test_serve_floor_boards_first_come_first_served():
    sim = make_sim(lift_capacity=2)
    state = sim.state()
    state.floor_passengers[0] = deque([Passenger(0, 1), Passenger(0, 2), Passenger(0, 3)])

    served, boarded = sim.serve_floor()

    assert (served, boarded) == (0, 2)
    assert [p.destination for p in state.lift_passengers] == [1, 2]
    assert list(state.floor_passengers[0]) == [Passenger(0, 3)]


def test_serve_floor_respects_capacity():
    sim = make_sim()
    for _ in range(200):
        sim.sample_passengers()
    for _ in range(sim.n_floors):
        sim.serve_floor()
        assert len(sim.state().lift_passengers) <= sim.lift_capacity
        sim.move_up()


def test_render_draws_every_floor(capsys):
    sim = make_sim()
    for _ in range(10):
        sim.sample_passengers()
    sim.render()
    assert capsys.readouterr().out.count("|") == sim.n_floors
