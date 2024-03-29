from unittest import mock

import pytest
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


def test_adding_system_twice():
    sim = Sim()
    empty_system = EmptyTestSystem()
    sim.add_system(empty_system)
    with pytest.raises(ValueError):
        sim.add_system(empty_system)


def test_observer_signature():
    sim = Sim()

    s = VanDerPol()
    sim.add_system(s)

    obs = mock.Mock()
    sim.add_observer(obs)

    sim.simulate(1, 1)

    args, _ = obs.call_args
    t, y = args

    assert t == 1
    assert np.array_equal(y, np.array([s.states.x, s.states.y]).flatten())


def test_observer_breaking():
    sim = Sim()

    s = VanDerPol()
    sim.add_system(s)

    sim.add_observer(lambda t, y: True if t > 5 else False)

    t_end = sim.simulate(10, 0.1)

    assert t_end == pytest.approx(5.1)


def test_observer_removal():
    sim = Sim()

    s = VanDerPol()
    sim.add_system(s)

    obs = mock.Mock(return_value=False)
    remover = sim.add_observer(obs)

    sim.simulate(3, 0.1)

    remover()

    t_end = sim.simulate(10, 0.1)

    assert t_end == pytest.approx(13)
    assert obs.call_count == 31


def test_add_observer_twice():
    sim = Sim()

    def noop():
        pass

    sim.add_observer(noop)
    with pytest.raises(ValueError):
        sim.add_observer(noop)


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
