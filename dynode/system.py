"""
Solving initial value problems of sets of connected and/or recursive
 dynamical systems through numerical integration.
"""

from copy import deepcopy
from functools import partial
from abc import ABC, abstractmethod

import numpy as np

from .containers import (ParameterContainer, VariableContainer,
                         ResultContainer)

class SystemInterface(ABC):

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
        return self._states

    @property
    def ders(self):
        return self._ders

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def res(self):
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
    def add_subsystem(self, sub):
        if sub is self:
            raise ValueError('Cant have self as subsystem to self!')

        if not sub in self._subs:
            self._subs.append(sub)

    # Connection API
    def add_pre_connection(self, connection_func):
        if connection_func in self._pre_connections:
            raise ValueError('This pre-connection has already been added!')

        self._pre_connections.append(connection_func)
        return partial(self._pre_connections.remove, connection_func)

    def add_post_connection(self, connection_func):
        if connection_func in self._post_connections:
            raise ValueError('This post-connection has already been added!')

        self._post_connections.append(connection_func)
        return partial(self._post_connections.remove, connection_func)

    # Results API
    def store(self, time):

        for sub in self._subs:
            sub.store(time)

        for prop, key in self._store_vars:
            val = deepcopy(getattr(self, prop)[key])
            self.res.store(key, val)

        self.res.store('time', time)

    def add_store(self, prop, name):
        self._store_vars.add((prop, name))

    # pylint: disable=protected-access
    def _step(self, time):
        # Apply pre-connections
        for con in self._pre_connections:
            con(self, time)

        # Recurse over subsystems
        for sub in self._subs:
            sub._step(time)

        # Step this system
        self.do_step(time)

        # Apply post-connections
        for con in self._post_connections:
            con(self, time)

    @abstractmethod
    def do_step(self, time):
        pass
