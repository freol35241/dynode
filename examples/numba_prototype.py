"""
Prototype: Numba Acceleration for Dynode Systems

This example demonstrates how to accelerate dynode systems using Numba's JIT compiler
with minimal code changes.

Performance expectations:
- Simple systems (VanDerPol): 2-3x speedup
- Complex systems (many states): 5-20x speedup
- First run: compilation overhead (~0.2-2s)
- Subsequent runs: full speedup realized
"""

import numpy as np
from numba import njit
import time

# Import dynode (assuming it's available)
try:
    from dynode import SystemInterface, Simulation, Recorder
except ImportError:
    print("Warning: dynode not found. This is a prototype only.")
    # Mock classes for demonstration
    class SystemInterface:
        def __init__(self):
            class Container:
                pass
            self.states = Container()
            self.ders = Container()
            self.inputs = Container()

    class Simulation:
        pass

    class Recorder:
        pass


# ==============================================================================
# Example 1: VanDerPol Oscillator - Plain vs Numba
# ==============================================================================

class VanDerPolPlain(SystemInterface):
    """Standard implementation without Numba"""

    def __init__(self):
        super().__init__()
        self.states.x = 0.0
        self.states.y = 1.0
        self.ders.dx = 0.0
        self.ders.dy = 0.0
        self.inputs.mu = 1.0

    def do_step(self, time):
        mu = self.inputs.mu
        x = self.states.x
        y = self.states.y

        self.ders.dx = y
        self.ders.dy = mu * (1 - x**2) * y - x


class VanDerPolNumba(SystemInterface):
    """Numba-accelerated implementation"""

    def __init__(self):
        super().__init__()
        self.states.x = 0.0
        self.states.y = 1.0
        self.ders.dx = 0.0
        self.ders.dy = 0.0
        self.inputs.mu = 1.0

    @staticmethod
    @njit
    def _compute_derivatives(x, y, mu):
        """Pure numerical function - compiled by Numba"""
        dx = y
        dy = mu * (1 - x**2) * y - x
        return dx, dy

    def do_step(self, time):
        self.ders.dx, self.ders.dy = self._compute_derivatives(
            self.states.x, self.states.y, self.inputs.mu
        )


# ==============================================================================
# Example 2: Complex System with Multiple States
# ==============================================================================

class ComplexSystemPlain(SystemInterface):
    """10-state system without Numba"""

    def __init__(self):
        super().__init__()
        # 10 states (e.g., coupled oscillators)
        for i in range(5):
            setattr(self.states, f'x{i}', 0.0)
            setattr(self.states, f'v{i}', 1.0)
            setattr(self.ders, f'dx{i}', 0.0)
            setattr(self.ders, f'dv{i}', 0.0)

        self.inputs.k = 1.0  # Spring constant
        self.inputs.c = 0.1  # Coupling

    def do_step(self, time):
        k = self.inputs.k
        c = self.inputs.c

        # Compute derivatives for coupled oscillators
        for i in range(5):
            x = getattr(self.states, f'x{i}')
            v = getattr(self.states, f'v{i}')

            # Coupling term
            coupling = 0.0
            if i > 0:
                coupling += c * (getattr(self.states, f'x{i-1}') - x)
            if i < 4:
                coupling += c * (getattr(self.states, f'x{i+1}') - x)

            setattr(self.ders, f'dx{i}', v)
            setattr(self.ders, f'dv{i}', -k * x + coupling)


class ComplexSystemNumba(SystemInterface):
    """10-state system with Numba"""

    def __init__(self):
        super().__init__()
        # Store as arrays for efficient Numba processing
        self.states.x = np.zeros(5)
        self.states.v = np.ones(5)
        self.ders.dx = np.zeros(5)
        self.ders.dv = np.zeros(5)

        self.inputs.k = 1.0
        self.inputs.c = 0.1

    @staticmethod
    @njit
    def _compute_derivatives(x, v, k, c):
        """Compiled computation of coupled oscillators"""
        n = len(x)
        dx = v.copy()  # Velocities
        dv = np.zeros(n)

        for i in range(n):
            # Coupling term
            coupling = 0.0
            if i > 0:
                coupling += c * (x[i-1] - x[i])
            if i < n - 1:
                coupling += c * (x[i+1] - x[i])

            dv[i] = -k * x[i] + coupling

        return dx, dv

    def do_step(self, time):
        self.ders.dx, self.ders.dv = self._compute_derivatives(
            self.states.x, self.states.v, self.inputs.k, self.inputs.c
        )


# ==============================================================================
# Example 3: System with Complex Math (Trigonometry)
# ==============================================================================

class PendulumPlain(SystemInterface):
    """Nonlinear pendulum without Numba"""

    def __init__(self):
        super().__init__()
        self.states.theta = 0.1
        self.states.omega = 0.0
        self.ders.dtheta = 0.0
        self.ders.domega = 0.0

        self.inputs.g = 9.81
        self.inputs.L = 1.0
        self.inputs.b = 0.1  # Damping

    def do_step(self, time):
        theta = self.states.theta
        omega = self.states.omega
        g = self.inputs.g
        L = self.inputs.L
        b = self.inputs.b

        self.ders.dtheta = omega
        self.ders.domega = -(g/L) * np.sin(theta) - b * omega


class PendulumNumba(SystemInterface):
    """Nonlinear pendulum with Numba (great for trig functions)"""

    def __init__(self):
        super().__init__()
        self.states.theta = 0.1
        self.states.omega = 0.0
        self.ders.dtheta = 0.0
        self.ders.domega = 0.0

        self.inputs.g = 9.81
        self.inputs.L = 1.0
        self.inputs.b = 0.1

    @staticmethod
    @njit
    def _compute_derivatives(theta, omega, g, L, b):
        """Numba compiles trig functions efficiently"""
        dtheta = omega
        domega = -(g/L) * np.sin(theta) - b * omega
        return dtheta, domega

    def do_step(self, time):
        self.ders.dtheta, self.ders.domega = self._compute_derivatives(
            self.states.theta, self.states.omega,
            self.inputs.g, self.inputs.L, self.inputs.b
        )


# ==============================================================================
# Benchmarking Utilities
# ==============================================================================

def benchmark_system(system, n_iterations=10000, warmup=True):
    """
    Benchmark a system's do_step() performance.

    Args:
        system: SystemInterface instance
        n_iterations: Number of do_step calls
        warmup: If True, run once first (for JIT compilation)

    Returns:
        Time in milliseconds
    """
    if warmup:
        system.do_step(0.0)  # Warmup for Numba compilation

    t0 = time.perf_counter()
    for i in range(n_iterations):
        system.do_step(float(i) * 0.01)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000  # Convert to ms


def compare_implementations(plain_class, numba_class, n_iterations=10000):
    """
    Compare plain vs Numba implementations.

    Returns:
        dict with timing results and speedup
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {plain_class.__name__} vs {numba_class.__name__}")
    print(f"{'='*60}")

    # Create instances
    plain = plain_class()
    numba = numba_class()

    # Benchmark plain implementation
    print("Running plain implementation...")
    time_plain = benchmark_system(plain, n_iterations, warmup=False)

    # Benchmark Numba (with compilation)
    print("Running Numba implementation (with compilation)...")
    time_numba_cold = benchmark_system(numba, n_iterations, warmup=False)

    # Benchmark Numba (without compilation)
    print("Running Numba implementation (pre-compiled)...")
    time_numba_warm = benchmark_system(numba, n_iterations, warmup=True)

    # Calculate speedups
    speedup_cold = time_plain / time_numba_cold
    speedup_warm = time_plain / time_numba_warm

    # Report
    print(f"\nResults ({n_iterations} iterations):")
    print(f"  Plain Python:          {time_plain:8.2f} ms")
    print(f"  Numba (cold start):    {time_numba_cold:8.2f} ms  ({speedup_cold:4.2f}x)")
    print(f"  Numba (warm):          {time_numba_warm:8.2f} ms  ({speedup_warm:4.2f}x)")

    if speedup_cold < 1.0:
        print(f"\n  ⚠️  Compilation overhead dominates for this system size")
        print(f"      Increase n_iterations or system complexity for benefit")
    elif speedup_warm > 5.0:
        print(f"\n  ✅ Excellent speedup! Numba is highly beneficial here")
    elif speedup_warm > 2.0:
        print(f"\n  ✅ Good speedup. Numba recommended for this system")
    else:
        print(f"\n  ⚠️  Modest speedup. Numba optional for this system")

    return {
        'plain': time_plain,
        'numba_cold': time_numba_cold,
        'numba_warm': time_numba_warm,
        'speedup_cold': speedup_cold,
        'speedup_warm': speedup_warm,
    }


# ==============================================================================
# Helper: Auto-generate Numba pattern
# ==============================================================================

def create_numba_decorator(state_names, input_names):
    """
    Factory to create Numba-compatible do_step pattern.

    This demonstrates how dynode could provide a helper to automate
    the Numba pattern.

    Example:
        @create_numba_decorator(['x', 'y'], ['mu'])
        def compute_vdp(x, y, mu):
            return y, mu * (1 - x**2) * y - x
    """
    def decorator(compute_func):
        compiled = njit(compute_func)

        def do_step(self, time):
            # Extract states
            states = [getattr(self.states, name) for name in state_names]
            inputs = [getattr(self.inputs, name) for name in input_names]

            # Call compiled function
            results = compiled(*states, *inputs)

            # Assign derivatives
            for i, name in enumerate(state_names):
                setattr(self.ders, f'd{name}', results[i])

        return do_step

    return decorator


# Example usage of helper:
class VanDerPolHelper(SystemInterface):
    """Using the helper decorator (cleanest pattern)"""

    def __init__(self):
        super().__init__()
        self.states.x = 0.0
        self.states.y = 1.0
        self.ders.dx = 0.0
        self.ders.dy = 0.0
        self.inputs.mu = 1.0

    @create_numba_decorator(['x', 'y'], ['mu'])
    def do_step(x, y, mu):
        """Pure computation - automatically JIT compiled"""
        dx = y
        dy = mu * (1 - x**2) * y - x
        return dx, dy


# ==============================================================================
# Main Demo
# ==============================================================================

if __name__ == '__main__':
    print("="*60)
    print("Dynode + Numba: Performance Prototype")
    print("="*60)

    # Benchmark 1: Simple system
    results1 = compare_implementations(
        VanDerPolPlain, VanDerPolNumba, n_iterations=10000
    )

    # Benchmark 2: Complex system
    results2 = compare_implementations(
        ComplexSystemPlain, ComplexSystemNumba, n_iterations=10000
    )

    # Benchmark 3: Trig-heavy system
    results3 = compare_implementations(
        PendulumPlain, PendulumNumba, n_iterations=10000
    )

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"VanDerPol:       {results1['speedup_warm']:.2f}x speedup (warm)")
    print(f"Complex System:  {results2['speedup_warm']:.2f}x speedup (warm)")
    print(f"Pendulum:        {results3['speedup_warm']:.2f}x speedup (warm)")

    avg_speedup = (results1['speedup_warm'] + results2['speedup_warm'] +
                   results3['speedup_warm']) / 3
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

    if avg_speedup > 3.0:
        print("\n✅ Recommendation: Integrate Numba into dynode")
    else:
        print("\n⚠️  Consider: Speedups modest, evaluate real-world use cases")

    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Run this prototype with actual dynode installation")
    print("  2. Test with representative user systems")
    print("  3. Measure full simulation time (not just do_step)")
    print("  4. If beneficial, implement dynode.numba_utils module")
    print("="*60)
