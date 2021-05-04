import pytest
import numpy as np

from test_systems import VanDerPol, SingleDegreeMass

from dynode import connect_signals, Simulation as Sim, Recorder


def test_VanDerPol(pinned):
    s = VanDerPol()
    s.states.x = 1

    rec = Recorder()
    rec.store(s, "states.x")

    sim = Sim()
    sim.add_system(s)
    sim.add_observer(rec)

    sim.simulate(100, 0.1)

    assert s.states.x == pinned.approx()
    assert s.states.y == pinned.approx()
    assert rec.results[s]["states.x"] == pinned.approx()


def test_connected_systems():
    rec = Recorder()

    sys1 = VanDerPol()
    sys1.states.x = 1
    rec.store(sys1, "states.y", "y")

    sys2 = VanDerPol()
    rec.store(sys2, "inputs.mu", "mu")

    # Simple connection
    sys1.add_post_connection(connect_signals(sys1.states, "y", sys2.inputs, "mu"))

    # Custom connection
    def custom_connection(sys, t):
        sys.inputs.mu = 0.1 * np.sin(0.1 * t)

    sys1.add_pre_connection(custom_connection)

    sim = Sim()
    sim.add_system(sys1)
    sim.add_system(sys2)
    sim.add_observer(rec)

    sim.simulate(10, 0.1)

    assert np.array_equal(rec.results[sys1]["y"][1:], rec.results[sys2]["mu"][1:])


def test_heirarchical_systems():
    rec = Recorder()

    sys1 = VanDerPol()
    sys1.states.x = 1
    rec.store(sys1, "states.y")

    sys11 = VanDerPol()
    sys11.states.x = 1
    rec.store(sys11, "states.y")

    sys2 = VanDerPol()
    sys2.states.x = 1
    rec.store(sys2, "inputs.mu")

    sys22 = VanDerPol()
    sys22.states.x = 1
    rec.store(sys22, "inputs.mu")

    # Simple connection
    sys1.add_post_connection(connect_signals(sys1.states, "y", sys2.inputs, "mu"))
    sys11.add_post_connection(connect_signals(sys11.states, "y", sys22.inputs, "mu"))

    # Custom connection
    def custom_connection(sys, t):
        sys.inputs.mu = 0.1 * np.sin(0.1 * t)

    sys1.add_pre_connection(custom_connection)
    sys11.add_pre_connection(custom_connection)

    sim = Sim()
    sim.add_system(sys1)
    sim.add_system(sys2)
    sys22.add_subsystem(sys11)
    sim.add_system(sys22)
    sim.add_observer(rec)

    sim.simulate(10, 0.1)

    assert np.array_equal(sys2.states.y, sys22.states.y)


def test_1DOF_MassSpring(pinned):
    s = SingleDegreeMass()
    s.inputs.mass = 10

    def spring(sys, t):
        sys.inputs.force = -10 * sys.states.x

    s.add_pre_connection(spring)

    # Initial state
    s.states.x = 10

    rec = Recorder()
    rec.store(s, "states.x", "x")

    sim = Sim()
    sim.add_system(s)
    sim.add_observer(rec)

    sim.simulate(20, 0.05)

    upper = max(rec.results[s]["x"][1:])
    lower = min(rec.results[s]["x"][1:])

    assert upper == pinned.approx(rel=0.001)
    assert lower == pinned.approx(rel=0.001)
