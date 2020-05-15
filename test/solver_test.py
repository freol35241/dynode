import pytest
import numpy as np

from test_systems import VanDerPol, SingleDegreeMass

from dynode import connect_signals
from dynode.simulation import Simulation as Sim

def test_VanDerPol():
    s = VanDerPol()
    sim = Sim()
    sim.add_system(s)

    sim.simulate(100, 0.1)

    assert(s.states.x.item() == pytest.approx(2.60973741))
    assert(s.states.y.item() == pytest.approx(0.55710081))

def test_connected_systems():
    sys1 = VanDerPol()
    sys1.add_store('states', 'y')

    sys2 = VanDerPol()
    sys2.add_store('inputs', 'b')

    # Simple connection
    sys1.add_post_connection(connect_signals(sys1.states, 'y', sys2.inputs, 'b'))

    # Custom connection
    def custom_connection(sys, t):
        sys.inputs.b = 0.1*np.sin(0.1*t)
    sys1.add_pre_connection(custom_connection)

    sim = Sim()
    sim.add_system(sys1)
    sim.add_system(sys2)

    sim.simulate(100, 0.1)

    assert(np.array_equal(sys1.res.y[1:], sys2.res.b[1:]))

def test_heirarchical_systems():

    sys1 = VanDerPol()
    sys1.add_store('states', 'y')

    sys11 = VanDerPol()
    sys11.add_store('states', 'y')

    sys2 = VanDerPol()
    sys2.add_store('inputs', 'b')

    sys22 = VanDerPol()
    sys22.add_store('inputs', 'b')

    # Simple connection
    sys1.add_post_connection(connect_signals(sys1.states, 'y', sys2.inputs, 'b'))
    sys11.add_post_connection(connect_signals(sys11.states, 'y', sys22.inputs, 'b'))

    # Custom connection
    def custom_connection(sys, t):
        sys.inputs.b = 0.1*np.sin(0.1*t)
    sys1.add_pre_connection(custom_connection)
    sys11.add_pre_connection(custom_connection)

    sim = Sim()
    sim.add_system(sys1)
    sim.add_system(sys2)
    sys22.add_subsystem(sys11)
    sim.add_system(sys22)

    sim.simulate(100, 0.1)

    assert(np.array_equal(sys1.states.y, sys11.states.y))


def test_1DOF_MassSpring():
    s = SingleDegreeMass()
    s.inputs.mass = 10

    def spring(sys, t):
        sys.inputs.force = -10*sys.states.x
    s.add_pre_connection(spring)

    # Initial state
    s.states.x = 10
    s.add_store('states', 'x')

    sim = Sim()
    sim.add_system(s)

    sim.simulate(20, 0.01)

    assert(max(s.res.x[1:]) == pytest.approx(10, 0.001))
    assert(min(s.res.x[1:]) == pytest.approx(-10, 0.001))
