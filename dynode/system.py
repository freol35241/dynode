"""
Solving initial value problems of sets of connected and/or recursive
 dynamical systems through numerical integration.
"""


from abc import ABC, abstractmethod
from typing import Callable, Type, Optional, List

from graphlib import TopologicalSorter

import numpy as np

from .containers import ParameterContainer, VariableContainer


class SystemInterface(ABC):
    """Abstract Base Class (ABC) defining the System Interface.

    A child class must implement the `do_step` method.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        super().__init__()

        # Subsystems
        self._subs = list()

        # I/O
        self._states = VariableContainer()
        self._ders = VariableContainer()
        self._inputs = ParameterContainer()
        self._outputs = ParameterContainer()

        # Connections
        self._pre_connections = dict()
        self._post_connections = dict()

    # Properties
    @property
    def states(self) -> VariableContainer:
        """The states of the system."""
        return self._states

    @property
    def ders(self) -> VariableContainer:
        """The derivatives of the system."""
        return self._ders

    @property
    def inputs(self) -> ParameterContainer:
        """The inputs of the system."""
        return self._inputs

    @property
    def outputs(self) -> ParameterContainer:
        """The outputs of the system."""
        return self._outputs

    # State/der handling
    def get_states(self):

        states = [sub.get_states() for sub in self._subs]
        states.extend(list(self.states.values()))

        return np.concatenate(states, axis=None) if states else np.array(states)

    def dispatch_states(self, idx, states):

        for sub in self._subs:
            idx = sub.dispatch_states(idx, states)

        for key, value in self.states.items():
            jdx = idx + value.size
            self.states[key][:] = states[idx:jdx].reshape(value.shape)
            idx = jdx

        return idx

    def get_ders(self, idx, ders):

        for sub in self._subs:
            idx = sub.get_ders(idx, ders)

        for value in self.ders.values():
            jdx = idx + value.size
            ders[idx:jdx] = value.flatten()
            idx = jdx

        return idx

    # Subsystem API
    def add_subsystem(self, sub_system: Type["SystemInterface"]) -> None:
        """Add a subsystem to this system

        Args:
            sub_system (Type[SystemInterface]): The subsystem to add

        Raises:
            ValueError: If the subsystem is `self`
        """
        if sub_system is self:
            raise ValueError("Cant have self as subsystem to self!")

        if not sub_system in self._subs:
            self._subs.append(sub_system)

    # Connection API
    def add_pre_connection(
        self, connection_func: Callable, dependees: Optional[List[Callable]] = None
    ) -> Callable:
        """Add a pre-connection to the system.

        `connection_func` is a callable of the form:

        .. highlight:: python
        .. code-block:: python

            def connection_func(system : SystemInterface, time : int):
                pass

        Args:
            connection_func (Callable): The connection function to add
            dependees (Optional[List[Callable]], optional): A list of dependees that
                `connection_func` depends on. A topological sort will be performed to
                ensure all pre-connections are executed in the correct order. Defaults
                to None.

        Raises:
            ValueError: If `connection_func` has already been added.

        Returns:
            Callable: A de-registering function. By calling this, `connection_func`
                will be removed from the list of pre-connections.
        """

        if connection_func in self._pre_connections:
            raise ValueError("This pre-connection has already been added!")

        dependees = dependees or getattr(connection_func, "dependees", [])

        self._pre_connections[connection_func] = dependees

        def _deleter():
            del self._pre_connections[connection_func]

        return _deleter

    def add_post_connection(
        self, connection_func: Callable, dependees: Optional[List[Callable]] = None
    ) -> Callable:
        """Add a post-connection to the system.

        `connection_func` is a callable of the form:

        .. highlight:: python
        .. code-block:: python

            def connection_func(system : SystemInterface, time : int):
                pass

        Args:
            connection_func (Callable): The connection function to add
            dependees (Optional[List[Callable]], optional): A list of dependees that
                `connection_func` depends on. A topological sort will be performed to
                ensure all post-connections are executed in the correct order. Defaults
                to None.

        Raises:
            ValueError: If `connection_func` has already been added.

        Returns:
            Callable: A de-registering function. By calling this, `connection_func`
                will be removed from the list of post-connections.
        """
        if connection_func in self._post_connections:
            raise ValueError("This post-connection has already been added!")

        dependees = dependees or getattr(connection_func, "dependees", [])

        self._post_connections[connection_func] = dependees

        def _deleter():
            del self._post_connections[connection_func]

        return _deleter

    # pylint: disable=protected-access
    def _step(self, time):
        # Recurse over subsystems
        for sub in self._subs:
            sub._step(time)

        # Apply pre-connections

        for con in TopologicalSorter(self._pre_connections).static_order():
            con(self, time)

        # Step this system
        self.do_step(time)

        # Apply post-connections
        for con in TopologicalSorter(self._post_connections).static_order():
            con(self, time)

    @abstractmethod
    def do_step(self, time: float):
        """Abstract method to be implemented by child classes!

        Args:
            time (float): Current simulation time
        """
