"""
Collection of simple systems for test purposes
"""
from dynode import SystemInterface

class VanDerPol(SystemInterface):

    def __init__(self):
        super().__init__()
        self.inputs.a = 1.0
        self.inputs.b = 1.0
        self.states.x = 1.0
        self.ders.dx = 0.0
        self.states.y = 0.0
        self.ders.dy = 0.0

    def do_step(self, time):
        a = self.inputs.a
        b = self.inputs.b
        x = self.states.x
        y = self.states.y

        self.ders.dx = a*x*(b-y*y)-y
        self.ders.dy = x

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
        self.ders.du = f/m

class EmptyTestSystem(SystemInterface):
    """Dummy test system 
    """
    def do_step(self, t):
        pass

class CompositeTestSystem(EmptyTestSystem):
    """Composite test system
    """
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