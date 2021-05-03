"""
Generic recording functionality
"""

from copy import deepcopy
from operator import attrgetter
from collections import defaultdict
from typing import Optional, Type

from .system import SystemInterface


class Recorder(defaultdict):
    """Generic recorder functor.

    Can be used as an `observer` in conjunction with a `Simulation`.
    """

    def __init__(self):
        super().__init__(lambda: defaultdict(list))
        self._stores = defaultdict(list)

    def store(
        self, system: Type[SystemInterface], attribute: str, alias: Optional[str] = None
    ) -> None:
        """Register a variable/parameter of a system for storing.

        Args:
            system (Type[SystemInterface]): System that owns this variable/parameter
            attribute (str): A string of the form `x.y.z` pointing to the
                variable/parameter to be stored
            alias (Optional[str], optional): A string under which the stored
                attribute will be available at. Defaults to None.

        Raises:
            AttributeError: If the provided `attribute` does not exist on `system`
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
