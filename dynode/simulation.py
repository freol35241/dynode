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

    def __init__(self):
        self._systems = list()
        self._events = set()

    def add_system(self, sys):
        if not sys in self._systems:
            self._systems.append(sys)

    def add_event(self, evt):
        self._events.add(evt)
        return partial(self._events.remove, evt)

    #pylint: disable=invalid-name, protected-access
    def simulate(self, t, store_dt, t0=0, integrator='dopri5', **kwargs):
        if not self._systems:
            raise RuntimeError('Need at least 1 system in the simulation!')

        # Initial state
        y0 = collect_states(self._systems)
        if not y0.size > 0:
            raise RuntimeError('Need at least one state/der combination!')

        dy = np.zeros_like(y0)

        # Systems of ODEs as single func
        def func(t, y):
            dispatch_states(y, self._systems)

            for sys in self._systems:
                sys._step(t)

            collect_ders(dy, self._systems)
            return dy

        # Setup of solver
        solver = ode(func)
        solver.set_integrator(integrator, **kwargs)
        solver.set_initial_value(y0, t=t0)

        # Store initial results
        for sys in self._systems:
            sys.store(t0)

        # Integrate
        steps = int(t/store_dt)

        for _ in range(steps):

            # Step
            solver.integrate(solver.t+store_dt)
            if not solver.successful():
                raise RuntimeError('Solver failed')

            # Store results
            for sys in self._systems:
                sys.store(solver.t)

            # Check events
            terminate = False
            for evt in self._events:
                if evt(solver.t, solver.y):
                    terminate = True
                    break

            # Bail if any event is True
            if terminate:
                break

        return solver.t
