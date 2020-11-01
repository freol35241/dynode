"""
Solving initial value problems of sets of connected and/or recursive
 dynamical systems through numerical integration.
"""

from copy import deepcopy
from functools import partial
from abc import ABC, abstractmethod
from typing import Callable
from operator import attrgetter

import numpy as np

from .containers import (ParameterContainer, VariableContainer,
                         ResultContainer)

class SystemInterface(ABC):
    """
    Abstract Base Class (ABC) defining the System Interface.

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
        self._pre_connections = list()
        self._post_connections = list()

        # Result
        self._store_vars = set()
        self._res = ResultContainer()

    # Properties
    @property
    def states(self):
        """
        VariableContainer of states.
        
        Accessible as attributes.
        """
        return self._states

    @property
    def ders(self):
        """
        VariableContainer of ders.
        
        Accessible as attributes.
        """
        return self._ders

    @property
    def inputs(self):
        """
        ParameterContainer of inputs.
        
        Accessible as attributes.
        """
        return self._inputs

    @property
    def outputs(self):
        """
        ParameterContainer of outputs.
        
        Accessible as attributes.
        """
        return self._outputs

    @property
    def res(self):
        """
        ResultContainer of stored results.
        
        Accessible as keys.
        """
        return self._res

    # State/der handling
    def get_states(self):

        states = [sub.get_states() for sub in self._subs]
        states.append(list(self.states.values()))

        return np.concatenate(
            states,
            axis=None
        )

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
    def add_subsystem(self, sub_system) -> None:
        """
        Add a subsystem to this system.
        
        Raises `ValueError` if sub is a reference to this system.
        """
        if sub_system is self:
            raise ValueError('Cant have self as subsystem to self!')

        if not sub_system in self._subs:
            self._subs.append(sub_system)

    # Connection API
    def add_pre_connection(self, connection_func) -> Callable:
        """
        Adds a pre-connection callable to this system.
        
        `connection_func` is a callable of the form:
        ```
        def connection_func(system : SystemInterface, time : int):
            pass
        ```
        
        Returns a callable that, when called, removes this pre-connection.
        """
        if connection_func in self._pre_connections:
            raise ValueError('This pre-connection has already been added!')

        self._pre_connections.append(connection_func)
        return partial(self._pre_connections.remove, connection_func)

    def add_post_connection(self, connection_func) -> Callable:
        """
        Adds a post-connection callable to this system.
        
        `connection_func` is a callable of the form:
        ```
        def connection_func(system : SystemInterface, time : int):
            pass
        ```
        
        Returns a callable that, when called, removes this post-connection.
        """
        if connection_func in self._post_connections:
            raise ValueError('This post-connection has already been added!')

        self._post_connections.append(connection_func)
        return partial(self._post_connections.remove, connection_func)

    # Results API
    def store(self, time):

        for sub in self._subs:
            sub.store(time)

        for attr_str, key_str in self._store_vars:
            val = deepcopy(attrgetter(attr_str)(self))
            self.res.store(key_str, val)

        self.res.store('time', time)

    def add_store(self, attribute : str, alias=None) -> None:
        """
        Adds an attribute or subattribute of this system
         to be stored during a simulation.
         
        `attribute` is a string of the form `x.y.z`
        
        `alias` is optionally a string under which the stored attribute will be available at in the result.
        
        Raises `AttributeError` if attribute is non-existing
        """
        attrgetter(attribute)(self) # Try to access attribute, raises AttributeError if non-existing
        self._store_vars.add((attribute, alias or attribute))

    # pylint: disable=protected-access
    def _step(self, time):
        # Recurse over subsystems
        for sub in self._subs:
            sub._step(time)
            
        # Apply pre-connections
        for con in self._pre_connections:
            con(self, time)

        # Step this system
        self.do_step(time)

        # Apply post-connections
        for con in self._post_connections:
            con(self, time)

    @abstractmethod
    def do_step(self, time):
        """
        To be implemented by child classes!
        """
        pass
