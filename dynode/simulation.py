"""
Solving initial value problems of sets of connected and/or recursive
 dynamical systems through numerical integration.
"""

from functools import partial

import numpy as np
from scipy.integrate import ode

def collect_states(systems):
    return np.concatenate(
        [sys.get_states() for sys in systems],
        axis=None
    )

def dispatch_states(states, systems):
    idx = 0
    for sys in systems:
        idx = sys.dispatch_states(idx, states)

    if len(states) != idx:
        raise RuntimeError('Mismatch in number of states and ders!')

def collect_ders(ders, systems):
    idx = 0
    for sys in systems:
        idx = sys.get_ders(idx, ders)

    if len(ders) != idx:
        raise RuntimeError('Mismatch in number of states and ders!')

class Simulation:
    """
    Simulation class representing a set of connected or unconnected
     dynamical systems that can be stepped forward in time by a
     numerical integration scheme.
    """
    def __init__(self):
        self._systems = list()
        self._events = set()
        self._t = 0
        
    @property
    def systems(self):
        """
        List of systems added to this simulation
        """
        return self._systems

    def add_system(self, system) -> None:
        """
        Adds a system to the simulation
        """
        if not system in self._systems:
            self._systems.append(system)

    def add_event(self, event) -> None:
        """
        Adds an event to this simulation.
        
        `event` is a callable of the form:
        ```
        def event(t : int, states : np.ndarray) -> bool:
            return True or False
        ```
        The simulation will break early if any such event returns True
        """
        self._events.add(event)
        return partial(self._events.remove, event)

    #pylint: disable=invalid-name, protected-access
    def simulate(self, t, store_dt, integrator='dopri5', **kwargs) -> int:
        """
        Step forward in time, `t` seconds while storing any stored variables and
         checking events every `store_dt` interval.
        
        Returns the current time of the simulation.
         
        Raise RuntimeErrors if:
        
        * There are no systems added to the simulation
        * There are no states/ders to be integrated
        * The solver fails due to numerical instabilities
        """
        if not self.systems:
            raise RuntimeError('Need at least 1 system in the simulation!')

        # Initial state
        y0 = collect_states(self._systems)
        if not y0.size > 0:
            raise RuntimeError('Need at least one state/der combination!')

        dy = np.zeros_like(y0)

        # Systems of ODEs as single func
        def func(t, y):
            dispatch_states(y, self.systems)

            for sys in self.systems:
                sys._step(t)

            collect_ders(dy, self.systems)
            return dy

        # Setup of solver
        solver = ode(func)
        solver.set_integrator(integrator, **kwargs)
        solver.set_initial_value(y0, t=self._t)

        # Store initial results
        if self._t == 0:
            for sys in self.systems:
                sys.store(self._t)

        # Integrate
        steps = int(t/store_dt)

        for _ in range(steps):

            # Step
            solver.integrate(solver.t+store_dt)
            if not solver.successful():
                raise RuntimeError('Solver failed')

            # Store results
            for sys in self.systems:
                sys.store(solver.t)

            # Check events
            terminate = False
            for event in self._events:
                if event(solver.t, solver.y):
                    terminate = True
                    break

            # Bail if any event is True
            if terminate:
                break
            
        # Update internal state
        self._t = solver.t

        return solver.t
