"""
Collection of simple systems for test purposes
"""
import mock
from dynode import SystemInterface


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
        self.ders.dy = mu * (1 - x ** 2) * y - x


class SingleDegreeMass(SystemInterface):
    def __init__(self):
        super().__init__()
        self.states.x = 0
        self.states.u = 0
        self.ders.dx = 0
        self.ders.du = 0
        self.inputs.force = 0
        self.inputs.mass = 0

    def do_step(self, time):
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
