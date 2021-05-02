import pytest
import mock
import numpy as np

from dynode import Simulation as Sim

from test_systems import EmptyTestSystem, VanDerPol, ErrorTestSystem, MockVanDerPol


def test_empty_simulation():
    sim = Sim()

    with pytest.raises(RuntimeError):
        sim.simulate(100, 0.1)


def test_simulation_with_no_states():
    sim = Sim()
    sim.add_system(EmptyTestSystem())

    with pytest.raises(RuntimeError):
        sim.simulate(100, 0.1)


def test_event_signature():
    sim = Sim()

    s = VanDerPol()
    sim.add_system(s)

    evt = mock.Mock()
    sim.add_event(evt)

    sim.simulate(1, 1)

    t, y = evt.call_args.args

    assert t == 1
    assert np.array_equal(y, np.array([s.states.x, s.states.y]).flatten())


def test_event_breaking():
    sim = Sim()

    s = VanDerPol()
    sim.add_system(s)

    sim.add_event(lambda t, y: True if t > 5 else False)

    t_end = sim.simulate(10, 0.1)

    assert t_end == pytest.approx(5.1)


def test_event_removal():
    sim = Sim()

    s = VanDerPol()
    sim.add_system(s)

    evt = mock.Mock(return_value=False)
    remover = sim.add_event(evt)

    sim.simulate(3, 0.1)

    remover()

    t_end = sim.simulate(10, 0.1)

    assert t_end == pytest.approx(13)
    assert evt.call_count == 30


def test_states_ders_mismatch():
    sim = Sim()

    sim.add_system(ErrorTestSystem())

    with pytest.raises(RuntimeError):
        sim.simulate(10, 0.1)


def test_fixed_step_size():
    sim = Sim()

    s = MockVanDerPol()
    s.states.x = 1
    sim.add_system(s)

    sim.simulate(3, 0.1, fixed_step=True)

    assert s.mock.call_count == 3 / 0.1 * 7
