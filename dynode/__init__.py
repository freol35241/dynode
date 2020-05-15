"""
dynode

A framework for modelling and simulation of dynamical systems in the form of
 ordinary differential equations.

"""

from .simulation import Simulation
from .system import SystemInterface

def connect_signals(cont1, key1, cont2, key2):
    # pylint: disable=unused-argument
    def connect(*args, **kwargs):
        cont2[key2] = cont1[key1]
    return connect
