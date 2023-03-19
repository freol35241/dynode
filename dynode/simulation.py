"""
Solving initial value problems of sets of connected and/or recursive
 dynamical systems through numerical integration.
"""

from functools import partial
from typing import List, Type, Callable

import numpy as np
from scipy.integrate import ode

from .system import SystemInterface


def collect_states(systems: List[SystemInterface]) -> np.ndarray:
    return np.concatenate([sys.get_states() for sys in systems], axis=None)


def dispatch_states(states: np.ndarray, systems: List[SystemInterface]):
    idx = 0
    for sys in systems:
        idx = sys.dispatch_states(idx, states)

    if len(states) != idx:
        raise RuntimeError("Mismatch in number of states and ders!")


def collect_ders(ders: np.ndarray, systems: List[SystemInterface]):
    idx = 0
    for sys in systems:
        idx = sys.get_ders(idx, ders)

    if len(ders) != idx:
        raise RuntimeError("Mismatch in number of states and ders!")


class Simulation:
    """A simulation.

    Representing a set of connected or unconnected dynamical systems that can be
    stepped forward in time by a numerical integration scheme.
    """

    def __init__(self):
        self._systems = []
        self._observers = []
        self._t = 0

    @property
    def systems(self) -> List[SystemInterface]:
        """Systems part of this simulation

        Returns:
            List[SystemInterface]: A list of the systems added to this simulation
        """
        return self._systems

    def add_system(self, system: Type[SystemInterface]) -> None:
        """Adds a system to the simulation.

        Args:
            system (Type[SystemInterface]): The system to add, adhering to the
                interface provided by `SystemInterface`.

        Raises:
            ValueError: If `system` has already been added to the simulation.
        """
        if system in self._systems:
            raise ValueError(
                f"System({system}) has already been added to this simulation!"
            )
        self._systems.append(system)

    def add_observer(self, observer: Callable) -> None:
        """Add an observer to the simulation.

        An `observer` is a callable of the form:

        .. highlight:: python
        .. code-block:: python

            def observer(t : int, states : np.ndarray) -> None:

        An `observer` can return True to signal that the simulation should break early:

        .. highlight:: python
        .. code-block:: python

            def observer(t : int, states : np.ndarray) -> bool:
                return True

        Args:
            observer (Callable): A callable that will be called every `observer_dt`.

        Raises:
            ValueError: If `observer` has already been added to the simulation.

        Returns:
            Callable: A de-registering function. By calling this, `observer`
                will be removed from the list of observers.
        """
        if observer in self._observers:
            raise ValueError(f"Observer({observer}) already registered!")

        self._observers.append(observer)
        return partial(self._observers.remove, observer)

    # pylint: disable=invalid-name, protected-access
    def simulate(
        self,
        t: float,
        observer_dt: float,
        fixed_step: bool = False,
        integrator: str = "dopri5",
        **kwargs,
    ) -> int:
        """Perform simulation.

        Step the collection of systems forward in time, `t` seconds while informing
        any `observer`s about the progress every `observer_dt` interval. If
        `fixed_step=True`, `observer_dt` is also used as the internal step size of the
        solver, leaving the user in charge of choosing a reasonable step size for the
        problem at hand.

        Args:
            t (float): Total time to progress
            observer_dt (float): Timestep with which the observers will be invoked.
            fixed_step (bool, optional): If True, `observer_dt` is used as the internal
                step size of the solver. Defaults to False.
            integrator (str, optional): Which solver to use, this argument is directly
                forwarded to `scipy.integrate.ode`. Defaults to "dopri5".

        Raises:
            RuntimeError: If there are no systems added to the simulation
            RuntimeError: If there are no states/ders to be integrated
            RuntimeError: The solver fails due to numerical instabilities

        Returns:
            int: The current time of the simulation.
        """
        if not self.systems:
            raise RuntimeError("Need at least 1 system in the simulation!")

        # Initial state
        y0 = collect_states(self._systems)
        if not y0.size > 0:
            raise RuntimeError("Need at least one state/der combination!")

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
        solver.set_initial_value(y0, t=self._t)

        # Try with a large step from the beginning
        kwargs.update({"first_step": observer_dt})

        # Fixed step size workaround
        if fixed_step:
            kwargs.update({"atol": np.inf})

        # Setup of integration scheme
        solver.set_integrator(integrator, **kwargs)

        # Inform the observers about the initial state
        if self._t == 0:
            for obs in self._observers:
                obs(self._t, solver.y)

        # Integrate
        steps = int(t / observer_dt)

        for _ in range(steps):
            # Step
            solver.integrate(solver.t + observer_dt)
            if not solver.successful():
                raise RuntimeError("Solver failed")

            # Inform observers and gather their responses
            terminate = False
            for obs in self._observers:
                if obs(solver.t, solver.y):
                    terminate = True
                    break

            # Bail if any observer returns True
            if terminate:
                break

        # Update internal state
        self._t = solver.t

        return solver.t
