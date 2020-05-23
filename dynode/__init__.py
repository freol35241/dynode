"""
A framework for modelling and simulation of dynamical systems in the form of
 ordinary differential equations.

.. include:: ../README.md
"""

__pdoc__ = {
    'containers': False,
    'simulation': False,
    'system': False,
    'SystemInterface.get_states': False,
    'SystemInterface.get_ders': False,
    'SystemInterface.dispatch_states': False,
    'SystemInterface.store': False
}

from .simulation import Simulation
from .system import SystemInterface
from typing import Callable

def connect_signals(container1, key1, container2, key2) -> Callable:
    """Returns callable that connects key1 of container1 to key2
     of container2. The callable can be used as a pre-step and/or
     post-step connection.
     
     Example usage, connecting state `x` on sys1 to input `a` on sys2:
     ```
     sys1.add_post_connection(
         connect_signals(sys1.states, 'x', sys2.inputs, 'a')
     )
     ```
    """    
    # pylint: disable=unused-argument
    def connect(*args, **kwargs):
        container2[key2] = container1[key1]
    return connect

__all__ = [
    'Simulation',
    'SystemInterface',
    'connect_signals'
]
