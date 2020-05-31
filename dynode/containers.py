"""Collection of data containers used by a dynode System
"""
from collections.abc import Sequence

import numpy as np

class ParameterContainer(dict):
    
    def __getattr__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)

    def __setattr__(self, *args, **kwargs):
        return self.__setitem__(*args, **kwargs)

class VariableContainer(ParameterContainer):

    def __setitem__(self, key, value):
        if np.isscalar(value):
            value = np.atleast_1d(value)

        if not isinstance(value, (Sequence, np.ndarray)):
            raise TypeError('States/Ders must be scalars, lists or np.ndarrays!')

        super().__setitem__(key, np.array(value, dtype=float))
        
    def __getattr__(self, *args, **kwargs):
        array = self.__getitem__(*args, **kwargs)
        
        try:
            return array.item()
        except ValueError:
            return array
        

class ResultContainer(dict):

    def store(self, key, value):
        """Store value associated with key.

        Parameters
        ----------
        key : str
            Key name associated with this value
        value : any
            Value to be stored
        """
        self[key] = self.get(key, [])
        self[key].append(value)
