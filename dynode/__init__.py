"""
A framework for modelling and simulation of dynamical systems in the form of
ordinary differential equations.
"""

from typing import Callable, Union

from .simulation import Simulation
from .recorder import Recorder
from .system import SystemInterface
from .containers import VariableContainer, ParameterContainer

AnyContainer = Union[VariableContainer, ParameterContainer]


def connect_signals(
    first: Union[VariableContainer, ParameterContainer],
    first_key: str,
    second: Union[VariableContainer, ParameterContainer],
    second_key: str,
) -> Callable:
    """Explicitly connect two variables/parameters.

    Returns a callable that connects `first_key` of `first` to `second_key` of
    `second`. The callable can be used as a pre-step and/or post-step connection.
    Connecting state `x` on `sys1` to input `a` on `sys2`:

    .. highlight:: python
    .. code-block:: python

        sys1.add_post_connection(
            connect_signals(sys1.states, "x", sys2.inputs, "a")
        )

    Args:
        first (Union[VariableContainer, ParameterContainer]): The container to
            connect from
        first_key (str): The variable name in `first` to connect from
        second (Union[VariableContainer, ParameterContainer]): The container to
            connect to
        second_key (str): The variable name in `second` to connect to

    Returns:
        Callable: A connection callable
    """

    # pylint: disable=unused-argument
    def connect(*args, **kwargs):
        setattr(second, second_key, getattr(first, first_key))

    return connect


__all__ = ["Simulation", "Recorder", "SystemInterface", "connect_signals"]
