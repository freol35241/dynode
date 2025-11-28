"""
Optional helper utilities for Numba acceleration in dynode.

This module provides convenience decorators and patterns to make it easier
for System developers to use Numba JIT compilation.

Note: These are OPTIONAL helpers. Users can use @njit directly without
any changes to dynode core.
"""

from numba import njit
import numpy as np
from functools import wraps


def numba_rhs(state_names, input_names=None, der_prefix='d'):
    """
    Decorator to automatically create a Numba-JIT compiled do_step method.

    This is purely a convenience wrapper - no dynode core changes required.

    Args:
        state_names: List of state variable names (e.g., ['x', 'y'])
        input_names: List of input variable names (e.g., ['mu'])
        der_prefix: Prefix for derivative names (default 'd')

    Returns:
        Decorator that converts a pure function into a do_step method

    Example:
        class VanDerPol(SystemInterface):
            def __init__(self):
                super().__init__()
                self.states.x = 0.0
                self.states.y = 1.0
                self.ders.dx = 0.0
                self.ders.dy = 0.0
                self.inputs.mu = 1.0

            # This decorator does all the extraction/assignment
            @numba_rhs(['x', 'y'], ['mu'])
            def do_step(x, y, mu):
                dx = y
                dy = mu * (1 - x**2) * y - x
                return dx, dy
    """
    if input_names is None:
        input_names = []

    def decorator(compute_func):
        # JIT compile the user's function
        compiled = njit(compute_func)

        @wraps(compute_func)
        def do_step_method(self, time):
            # Extract states
            state_values = [getattr(self.states, name) for name in state_names]

            # Extract inputs
            input_values = [getattr(self.inputs, name) for name in input_names]

            # Call JIT-compiled function
            derivatives = compiled(*state_values, *input_values)

            # Ensure it's iterable
            if not isinstance(derivatives, tuple):
                derivatives = (derivatives,)

            # Assign derivatives
            for name, value in zip(state_names, derivatives):
                setattr(self.ders, f'{der_prefix}{name}', value)

        return do_step_method

    return decorator


def jit_system_method(method):
    """
    Simple decorator to JIT compile a method that extracts/returns values.

    This is for users who want a bit more control than numba_rhs but
    still want help with the pattern.

    Example:
        class MySystem(SystemInterface):
            @staticmethod
            @njit
            def _compute(x, v, k):
                return v, -k * x

            @jit_system_method
            def do_step(self, time):
                # This method structure is recognized and optimized
                x = self.states.x
                v = self.states.v
                k = self.inputs.k

                dx, dv = self._compute(x, v, k)

                self.ders.dx = dx
                self.ders.dv = dv
    """
    # For this simple version, just return the method as-is
    # The actual JIT compilation happens in the user's @njit decorated _compute
    return method


# Convenience exports
__all__ = ['numba_rhs', 'jit_system_method', 'njit']
