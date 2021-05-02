"""
Generic recording functionality
"""

from copy import deepcopy
from operator import attrgetter
from collections import defaultdict


class Recorder(defaultdict):
    def __init__(self):
        super().__init__(lambda: defaultdict(list))
        self._stores = defaultdict(list)

    def store(self, system, attribute: str, alias: str = None) -> None:
        """
        Adds an attribute or subattribute of 'system'
         to be stored during a simulation.

        `attribute` is a string of the form `x.y.z`

        `alias` is optionally a string under which the stored attribute will be
         available at in the result.

        Raises `AttributeError` if attribute is non-existing
        """
        attrgetter(attribute)(
            system
        )  # Try to access attribute, raises AttributeError if non-existing

        self._stores[system].append((attribute, alias or attribute))

    def __call__(self, time, _):
        for sys, store_vars in self._stores.items():
            self[sys]["time"].append(time)
            for attr_str, key_str in store_vars:
                val = deepcopy(attrgetter(attr_str)(sys))
                self[sys][key_str].append(
                    val
                )  # TODO: Find a better way to handle `sys`
