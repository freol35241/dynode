"""
Tests for Numba-accelerated systems.

These tests verify that:
1. Numba-JIT compiled systems produce numerically equivalent results
2. The systems work correctly with dynode's Simulation
3. The pattern is compatible with all dynode features
"""

import pytest
import numpy as np

from dynode import Simulation, Recorder

# Import test systems
from test_systems import VanDerPol, NUMBA_AVAILABLE

# Skip all tests if Numba is not available
pytestmark = pytest.mark.skipif(
    not NUMBA_AVAILABLE, reason="Numba not installed (optional dependency)"
)

if NUMBA_AVAILABLE:
    from test_systems import VanDerPolNumba


class TestNumbaSystem:
    """Test Numba-accelerated VanDerPol system"""

    def test_numba_system_exists(self):
        """Verify VanDerPolNumba can be instantiated"""
        system = VanDerPolNumba()
        assert system is not None
        assert system.states.x == 0.0
        assert system.states.y == 0.0
        assert system.inputs.mu == 1.0

    def test_numba_do_step(self):
        """Verify do_step computes derivatives correctly"""
        system = VanDerPolNumba()

        # Set known state
        system.states.x = 1.0
        system.states.y = 0.5
        system.inputs.mu = 1.0

        # Compute derivatives
        system.do_step(0.0)

        # Check results
        assert system.ders.dx == 0.5
        # dy = mu * (1 - x^2) * y - x = 1.0 * (1 - 1) * 0.5 - 1.0 = -1.0
        assert system.ders.dy == -1.0

    def test_numerical_equivalence_with_plain_python(self):
        """
        Verify Numba version produces identical results to plain Python version.
        """
        plain = VanDerPol()
        numba = VanDerPolNumba()

        # Set same initial conditions
        plain.states.x = 0.5
        plain.states.y = 1.2
        plain.inputs.mu = 1.5

        numba.states.x = 0.5
        numba.states.y = 1.2
        numba.inputs.mu = 1.5

        # Compute derivatives
        plain.do_step(0.0)
        numba.do_step(0.0)

        # Compare results (should be identical within floating point precision)
        np.testing.assert_allclose(plain.ders.dx, numba.ders.dx, rtol=1e-15)
        np.testing.assert_allclose(plain.ders.dy, numba.ders.dy, rtol=1e-15)

    def test_simulation_with_numba_system(self):
        """Verify Numba system works with dynode Simulation"""
        sim = Simulation()
        system = VanDerPolNumba()

        # Set initial conditions
        system.states.x = 0.0
        system.states.y = 1.0

        sim.add_system(system)

        # Simulate
        final_time = sim.simulate(t=1.0, observer_dt=0.1)

        assert final_time == pytest.approx(1.0)
        # State should have evolved
        assert system.states.x != 0.0 or system.states.y != 1.0

    def test_recorder_with_numba_system(self):
        """Verify Recorder works with Numba system"""
        sim = Simulation()
        system = VanDerPolNumba()

        # Set initial conditions
        system.states.x = 0.0
        system.states.y = 1.0

        sim.add_system(system)

        # Add recorder
        rec = Recorder()
        rec.store(system, "states.x")
        rec.store(system, "states.y")
        sim.add_observer(rec)

        # Simulate
        sim.simulate(t=1.0, observer_dt=0.1)

        # Check recorder captured data
        assert len(rec.results[system]["states.x"]) == 11  # t=0 + 10 steps
        assert len(rec.results[system]["states.y"]) == 11

    def test_multiple_calls_work(self):
        """
        Verify multiple calls to do_step work (tests Numba compilation caching).
        """
        system = VanDerPolNumba()

        # First call (triggers compilation)
        system.states.x = 1.0
        system.states.y = 0.0
        system.do_step(0.0)
        first_dy = system.ders.dy

        # Second call (uses cached compilation)
        system.states.x = 1.0
        system.states.y = 0.0
        system.do_step(0.0)
        second_dy = system.ders.dy

        # Results should be identical
        assert first_dy == second_dy

    def test_full_simulation_equivalence(self):
        """
        Run full simulation with both plain and Numba versions,
        verify they produce the same trajectory.
        """
        # Plain Python version
        sim_plain = Simulation()
        plain = VanDerPol()
        plain.states.x = 0.0
        plain.states.y = 1.0
        plain.inputs.mu = 1.0
        sim_plain.add_system(plain)

        rec_plain = Recorder()
        rec_plain.store(plain, "states.x", "x")
        rec_plain.store(plain, "states.y", "y")
        sim_plain.add_observer(rec_plain)

        sim_plain.simulate(t=10.0, observer_dt=0.1)

        # Numba version
        sim_numba = Simulation()
        numba = VanDerPolNumba()
        numba.states.x = 0.0
        numba.states.y = 1.0
        numba.inputs.mu = 1.0
        sim_numba.add_system(numba)

        rec_numba = Recorder()
        rec_numba.store(numba, "states.x", "x")
        rec_numba.store(numba, "states.y", "y")
        sim_numba.add_observer(rec_numba)

        sim_numba.simulate(t=10.0, observer_dt=0.1)

        # Compare trajectories
        x_plain = np.array(rec_plain.results[plain]["x"])
        x_numba = np.array(rec_numba.results[numba]["x"])

        y_plain = np.array(rec_plain.results[plain]["y"])
        y_numba = np.array(rec_numba.results[numba]["y"])

        # Should be numerically equivalent
        np.testing.assert_allclose(x_plain, x_numba, rtol=1e-10)
        np.testing.assert_allclose(y_plain, y_numba, rtol=1e-10)
