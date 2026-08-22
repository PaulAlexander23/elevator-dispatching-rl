from sim import LiftSim, LiftState


def test_sim():
    sim = LiftSim()
    sim.state()


def test_state():
    state = LiftState(10)
    assert state.lift_position == 0
    assert state.lift_passengers == []
    assert len(state.floor_passengers) == 10


def test_sample():
    sim = LiftSim()
    for n in range(100):
        sim.sample_passengers()
    print(sim.state())


def test_move_up():
    sim = LiftSim()
    sim.move_up()
    print(sim.state())
    sim._state.lift_position = 9
    sim.move_up()
    print(sim.state())


def test_move_down():
    sim = LiftSim()
    sim._state.lift_position = 5
    sim.move_down()
    print(sim.state())
    sim._state.lift_position = 0
    sim.move_down()
    print(sim.state())


def test_serve_floor():
    sim = LiftSim()
    for n in range(100):
        sim.sample_passengers()
    print(sim.state())
    sim.serve_floor()
    print(sim.state())


def test_serve_floor_move_and_server_floor():
    sim = LiftSim()
    for n in range(100):
        sim.sample_passengers()
    print(sim.state())
    sim.serve_floor()
    print(sim.state())
    sim.move_up()
    sim.serve_floor()
    print(sim.state())


def test_render():
    sim = LiftSim()
    for n in range(10):
        sim.sample_passengers()
    sim.render()


if __name__ == "__main__":
    test_sim()
    test_state()
    test_sample()
    test_move_up()
    test_move_down()
    test_serve_floor()
    test_serve_floor_move_and_server_floor()
    test_render()
