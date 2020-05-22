"""Collection of data containers used by a dynode System
"""
from collections.abc import Sequence

import numpy as np

class _BaseContainer(dict):

    def __getattr__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)

class ParameterContainer(_BaseContainer):

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

class VariableContainer(ParameterContainer):

    def __setitem__(self, key, value):
        if np.isscalar(value):
            value = np.atleast_1d(value)

        if not isinstance(value, (Sequence, np.ndarray)):
            raise TypeError('States/Ders must be lists or np.ndarrays!')

        super().__setitem__(key, np.array(value, dtype=float))

class ResultContainer(_BaseContainer):

    def store(self, key, value):
        self[key] = self.get(key, [])
        self[key].append(value)
