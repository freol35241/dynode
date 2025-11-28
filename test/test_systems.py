"""
Collection of simple systems for test purposes
"""
from unittest import mock

from dynode import SystemInterface

# Optional: Numba for JIT compilation
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class VanDerPol(SystemInterface):
    def __init__(self):
        super().__init__()
        self.inputs.mu = 1.0
        self.states.x = 0.0
        self.ders.dx = 0.0
        self.states.y = 0.0
        self.ders.dy = 0.0

    def do_step(self, time):
        mu = self.inputs.mu
        x = self.states.x
        y = self.states.y

        self.ders.dx = y
        self.ders.dy = mu * (1 - x**2) * y - x


class SingleDegreeMass(SystemInterface):
    def __init__(self):
        super().__init__()
        self.states.x = 0
        self.states.u = 0
        self.ders.dx = 0
        self.ders.du = 0
        self.inputs.force = 0
        self.inputs.mass = 0

    def do_step(self, _):
        m = self.inputs.mass
        f = self.inputs.force

        self.ders.dx = self.states.u
        self.ders.du = f / m


class EmptyTestSystem(SystemInterface):
    """Dummy test system"""

    def do_step(self, t):
        pass


class CompositeTestSystem(EmptyTestSystem):
    """Composite test system"""

    def __init__(self):
        super().__init__()
        self.add_subsystem(VanDerPol())
        self.add_subsystem(SingleDegreeMass())


class ErrorTestSystem(EmptyTestSystem):
    def __init__(self):
        super().__init__()
        self.states.x = [1, 2]
        self.states.y = 0
        self.ders.dx = 0
        self.ders.dy = 0


class MockVanDerPol(VanDerPol):
    def __init__(self):
        super().__init__()
        self.mock = mock.Mock()

    def do_step(self, time):
        self.mock(time)
        super().do_step(time)


if NUMBA_AVAILABLE:
    class VanDerPolNumba(SystemInterface):
        """
        Van der Pol oscillator with Numba JIT compilation.

        This demonstrates how to use Numba to accelerate the numerical
        computation in do_step(). The pattern is:
        1. Extract numerical logic to a static method
        2. Decorate with @njit for JIT compilation
        3. Call from do_step()

        Expected speedup: 2-5x for this simple system after warmup.
        """

        def __init__(self):
            super().__init__()
            self.inputs.mu = 1.0
            self.states.x = 0.0
            self.ders.dx = 0.0
            self.states.y = 0.0
            self.ders.dy = 0.0

        @staticmethod
        @njit
        def _compute_derivatives(x, y, mu):
            """
            Pure numerical function compiled with Numba.

            This function:
            - Takes primitives (floats) as input
            - Returns primitives as output
            - Has no side effects
            - Perfect for Numba nopython mode
            """
            dx = y
            dy = mu * (1 - x**2) * y - x
            return dx, dy

        def do_step(self, time):
            """
            Extract values from containers, call JIT-compiled function,
            assign results back.
            """
            dx, dy = self._compute_derivatives(
                self.states.x, self.states.y, self.inputs.mu
            )
            self.ders.dx = dx
            self.ders.dy = dy
