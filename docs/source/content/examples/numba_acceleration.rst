.. dynode-numba-acceleration

Accelerating Systems with Numba
==========================================

For compute-intensive systems, you can use `Numba <https://numba.pydata.org/>`__
to JIT-compile the numerical computation in your ``do_step()`` method, achieving
**2-20x speedup** depending on system complexity.

Installation
------------

Numba is an optional dependency. Install it with:

.. code:: bash

    pip install numba

Overview
--------

Numba works best when you:

1. Extract the numerical computation to a separate function
2. Decorate it with ``@njit`` (JIT compilation)
3. Call it from ``do_step()``

This separates pure numerical computation (which Numba compiles) from
container access (which stays in Python).

Example: Van der Pol with Numba
--------------------------------

Standard Implementation
~~~~~~~~~~~~~~~~~~~~~~~

Here's the standard Van der Pol oscillator:

.. code:: python

    from dynode import SystemInterface

    class VanDerPol(SystemInterface):
        def __init__(self):
            super().__init__()
            self.inputs.mu = 1.0
            self.states.x = 0.0
            self.ders.dx = 0.0
            self.states.y = 0.0
            self.ders.dy = 0.0

        def do_step(self, time):
            mu = self.inputs.mu
            x = self.states.x
            y = self.states.y

            self.ders.dx = y
            self.ders.dy = mu * (1 - x**2) * y - x

Numba-Accelerated Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's the same system with Numba JIT compilation:

.. code:: python

    from dynode import SystemInterface
    from numba import njit

    class VanDerPolNumba(SystemInterface):
        def __init__(self):
            super().__init__()
            self.inputs.mu = 1.0
            self.states.x = 0.0
            self.ders.dx = 0.0
            self.states.y = 0.0
            self.ders.dy = 0.0

        @staticmethod
        @njit
        def _compute_derivatives(x, y, mu):
            """Pure numerical function - JIT compiled by Numba"""
            dx = y
            dy = mu * (1 - x**2) * y - x
            return dx, dy

        def do_step(self, time):
            # Extract values, call compiled function, assign results
            dx, dy = self._compute_derivatives(
                self.states.x, self.states.y, self.inputs.mu
            )
            self.ders.dx = dx
            self.ders.dy = dy

Usage
~~~~~

The Numba-accelerated system is used exactly like the standard version:

.. code:: python

    from dynode import Simulation, Recorder

    sys = VanDerPolNumba()

    rec = Recorder()
    rec.store(sys, 'states.x', alias='x')
    rec.store(sys, 'states.y', alias='y')

    sim = Simulation()
    sim.add_system(sys)
    sim.add_observer(rec)

    sys.states.x = 1
    sim.simulate(100, 0.1)

    import matplotlib.pyplot as plt
    plt.plot(rec[sys]['time'], rec[sys]['x'])
    plt.plot(rec[sys]['time'], rec[sys]['y'])
    plt.show()

Performance Considerations
--------------------------

First Call (Compilation)
~~~~~~~~~~~~~~~~~~~~~~~~

The first time you call ``do_step()``, Numba compiles the ``_compute_derivatives``
function. This takes ~0.1-2 seconds. Subsequent calls use the compiled code and
are much faster.

Expected Speedups
~~~~~~~~~~~~~~~~~

Speedup depends on system complexity:

- **Simple systems (2-5 states):** 1.5-3x speedup
- **Medium systems (10-20 states):** 3-8x speedup
- **Complex systems (50+ states):** 10-20x speedup
- **Systems with trigonometry/exponentials:** 5-15x speedup

When to Use Numba
~~~~~~~~~~~~~~~~~

Numba is most beneficial for:

- Systems with many states (>10)
- Complex mathematical operations (``sin``, ``exp``, ``sqrt``, etc.)
- Long simulations (>1000 integration steps)
- Repeated simulations (compilation overhead amortized)

Numba may not help for:

- Very simple systems (overhead dominates)
- Short simulations (compilation time not amortized)
- Systems that are already using efficient NumPy operations

Pattern for More Complex Systems
---------------------------------

For systems with many states, you can use arrays:

.. code:: python

    from dynode import SystemInterface
    from numba import njit
    import numpy as np

    class CoupledOscillators(SystemInterface):
        def __init__(self, n_oscillators=10):
            super().__init__()
            self.n = n_oscillators

            # Use arrays for efficient processing
            self.states.x = np.zeros(n_oscillators)
            self.states.v = np.ones(n_oscillators)
            self.ders.dx = np.zeros(n_oscillators)
            self.ders.dv = np.zeros(n_oscillators)

            self.inputs.k = 1.0  # Spring constant
            self.inputs.c = 0.1  # Coupling

        @staticmethod
        @njit
        def _compute_derivatives(x, v, k, c):
            """JIT-compiled computation for all oscillators"""
            n = len(x)
            dx = v.copy()
            dv = np.zeros(n)

            for i in range(n):
                # Coupling to neighbors
                coupling = 0.0
                if i > 0:
                    coupling += c * (x[i-1] - x[i])
                if i < n - 1:
                    coupling += c * (x[i+1] - x[i])

                dv[i] = -k * x[i] + coupling

            return dx, dv

        def do_step(self, time):
            self.ders.dx, self.ders.dv = self._compute_derivatives(
                self.states.x, self.states.v,
                self.inputs.k, self.inputs.c
            )

What Can Be JIT-Compiled
------------------------

Numba's ``@njit`` (nopython mode) supports:

**Supported:**

- NumPy arrays and operations
- Basic math: ``+``, ``-``, ``*``, ``/``, ``**``
- NumPy functions: ``np.sin``, ``np.exp``, ``np.sqrt``, etc.
- Control flow: ``if``, ``for``, ``while``
- Tuple/list creation (with known types)

**Not Supported:**

- Python dictionaries (use ``numba.typed.Dict`` instead)
- String operations
- Dynamic attribute access (``getattr``, ``setattr``)
- Most Python standard library

For a complete list, see the `Numba documentation <https://numba.pydata.org/numba-doc/latest/reference/pysupported.html>`__.

Troubleshooting
---------------

Compilation Errors
~~~~~~~~~~~~~~~~~~

If you get Numba compilation errors, ensure your ``_compute_derivatives`` function:

- Only uses Numba-supported operations
- Takes primitives (floats, ints, NumPy arrays) as input
- Returns primitives as output
- Has no side effects (doesn't modify external state)

Numerical Differences
~~~~~~~~~~~~~~~~~~~~~

Numba uses the same floating-point arithmetic as Python, but may optimize
differently. Results should be identical within numerical precision (``~1e-15``).

To verify, compare your Numba system against the plain Python version using the
same initial conditions.

Performance Not Improving
~~~~~~~~~~~~~~~~~~~~~~~~~

If you don't see speedup:

- **System too simple:** Overhead dominates for very simple computations
- **Simulation too short:** Compilation time not amortized
- **Already using NumPy:** NumPy operations are already compiled C code

Profile your code to identify bottlenecks (most time should be in ``do_step()``).

Summary
-------

**To accelerate a dynode system with Numba:**

1. Install Numba: ``pip install numba``
2. Extract numerical computation to a static method
3. Decorate with ``@staticmethod`` and ``@njit``
4. Call from ``do_step()``

**Expected speedup:** 2-20x depending on complexity

**No changes needed to dynode core** - Numba integration is entirely user-side!
