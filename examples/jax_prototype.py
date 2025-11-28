"""
Prototype: JAX Integration for Dynode Systems

This example demonstrates how a JAX-based dynode module would work,
with GPU acceleration, JIT compilation, and automatic differentiation.

Performance expectations:
- CPU: 2-10x speedup (similar to Numba)
- GPU: 10-100x speedup (large systems)
- Autodiff: Enable gradient-based parameter optimization
- Compilation: 2-30s first run, then very fast
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
except ImportError:
    print("JAX not installed. Install with: pip install jax jaxlib")
    JAX_AVAILABLE = False
    # Mock for demonstration
    jnp = np
    def jit(f): return f
    def grad(f): return f
    def vmap(f): return f

try:
    from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
    DIFFRAX_AVAILABLE = True
except ImportError:
    print("Diffrax not installed. Install with: pip install diffrax")
    DIFFRAX_AVAILABLE = False


# ==============================================================================
# Proposed JAX-based API for dynode
# ==============================================================================

class JaxSystemInterface:
    """
    Base class for JAX-accelerated dynode systems.

    Key differences from classic SystemInterface:
    1. Functional paradigm: dynamics() returns derivatives, doesn't mutate
    2. State/params are PyTrees (nested dicts/arrays)
    3. Pure functions enable JIT, grad, vmap
    """

    def dynamics(self, t, state, params):
        """
        Compute derivatives for the system.

        Args:
            t: float, current time
            state: PyTree (dict/array) of state variables
            params: PyTree (dict/array) of parameters

        Returns:
            PyTree with same structure as state, containing derivatives
        """
        raise NotImplementedError("Subclasses must implement dynamics()")


class JaxSimulation:
    """
    JAX-based simulation using Diffrax for ODE solving.

    Supports:
    - GPU/TPU acceleration
    - JIT compilation
    - Automatic differentiation
    - Vectorization (vmap)
    """

    def __init__(self, system, solver='Dopri5'):
        """
        Args:
            system: JaxSystemInterface instance
            solver: Diffrax solver name ('Dopri5', 'Tsit5', 'Euler', etc.)
        """
        self.system = system
        self.solver_name = solver

    def solve(self, t_span, y0, params, saveat=None, **solver_kwargs):
        """
        Solve the ODE system.

        Args:
            t_span: (t0, t1) tuple
            y0: Initial state (PyTree)
            params: Parameters (PyTree)
            saveat: jnp.array of times to save, or SaveAt object
            **solver_kwargs: Additional solver options

        Returns:
            Solution object with .ts (times) and .ys (states)
        """
        if not DIFFRAX_AVAILABLE:
            raise ImportError("Diffrax required. Install: pip install diffrax")

        # Wrap system dynamics for Diffrax
        def vector_field(t, y, args):
            return self.system.dynamics(t, y, args)

        # Create ODE term and solver
        term = ODETerm(vector_field)

        if self.solver_name == 'Dopri5':
            solver = Dopri5()
        else:
            raise ValueError(f"Solver {self.solver_name} not implemented")

        # Setup save times
        if saveat is None:
            saveat = SaveAt(t0=True, t1=True)
        elif isinstance(saveat, jnp.ndarray):
            saveat = SaveAt(ts=saveat)

        t0, t1 = t_span

        # Solve
        solution = diffeqsolve(
            term, solver,
            t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=params,
            saveat=saveat,
            **solver_kwargs
        )

        return solution


# ==============================================================================
# Example 1: VanDerPol Oscillator
# ==============================================================================

class VanDerPolJax(JaxSystemInterface):
    """Van der Pol oscillator in JAX"""

    def dynamics(self, t, state, params):
        """
        Functional implementation - no mutation!

        Args:
            state: dict with keys 'x', 'y'
            params: dict with key 'mu'

        Returns:
            dict with derivatives for 'x', 'y'
        """
        x = state['x']
        y = state['y']
        mu = params['mu']

        return {
            'x': y,
            'y': mu * (1 - x**2) * y - x
        }


# ==============================================================================
# Example 2: Composite System (Coupled Oscillators)
# ==============================================================================

class CoupledOscillatorsJax(JaxSystemInterface):
    """Two coupled Van der Pol oscillators"""

    def dynamics(self, t, state, params):
        """
        Demonstrates subsystem composition in JAX.

        State structure:
            {'osc1': {'x': ..., 'y': ...},
             'osc2': {'x': ..., 'y': ...}}
        """
        # Extract subsystem states
        osc1_state = state['osc1']
        osc2_state = state['osc2']

        # Extract parameters
        mu1 = params['osc1']['mu']
        mu2 = params['osc2']['mu']
        coupling = params['coupling']

        # Compute each oscillator
        x1, y1 = osc1_state['x'], osc1_state['y']
        x2, y2 = osc2_state['x'], osc2_state['y']

        # Coupling: osc2's mu depends on osc1's position
        mu2_coupled = mu2 + coupling * x1

        # Derivatives
        return {
            'osc1': {
                'x': y1,
                'y': mu1 * (1 - x1**2) * y1 - x1
            },
            'osc2': {
                'x': y2,
                'y': mu2_coupled * (1 - x2**2) * y2 - x2
            }
        }


# ==============================================================================
# Example 3: Array-based System (Efficient for many similar states)
# ==============================================================================

class OscillatorArrayJax(JaxSystemInterface):
    """N coupled oscillators using arrays for efficiency"""

    def __init__(self, n_oscillators=10):
        self.n = n_oscillators

    def dynamics(self, t, state, params):
        """
        State: {'x': array[n], 'v': array[n]}
        Params: {'k': float, 'c': float} (stiffness, coupling)
        """
        x = state['x']  # positions
        v = state['v']  # velocities
        k = params['k']
        c = params['c']

        # Compute coupling forces (nearest neighbor)
        coupling = jnp.zeros_like(x)
        coupling = coupling.at[1:-1].set(
            c * (x[:-2] - 2*x[1:-1] + x[2:])  # Discrete Laplacian
        )

        # Derivatives
        dx = v
        dv = -k * x + coupling

        return {'x': dx, 'v': dv}


# ==============================================================================
# Advanced Features: Automatic Differentiation
# ==============================================================================

def parameter_sensitivity(system, t_span, y0, params, param_name):
    """
    Compute sensitivity of final state to a parameter using autodiff.

    This is impossible with scipy/Numba, but trivial with JAX!

    Args:
        system: JaxSystemInterface
        t_span, y0, params: As in solve()
        param_name: Name of parameter to differentiate wrt

    Returns:
        Gradient of final state wrt parameter
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX required")

    sim = JaxSimulation(system)

    def final_state_value(param_value):
        """Function to differentiate: param -> final state"""
        # Update params with new value
        updated_params = params.copy()
        updated_params[param_name] = param_value

        # Solve
        sol = sim.solve(t_span, y0, updated_params)

        # Return some scalar output (e.g., final x value)
        if isinstance(sol.ys, dict):
            return sol.ys['x'][-1]  # Final x value
        else:
            return sol.ys[-1]  # Final state

    # Compute gradient using JAX autodiff
    grad_fn = grad(final_state_value)
    sensitivity = grad_fn(params[param_name])

    return sensitivity


# ==============================================================================
# Advanced Features: Vectorization (Batch Simulations)
# ==============================================================================

def batch_simulations(system, t_span, y0_batch, params_batch, saveat=None):
    """
    Run multiple simulations in parallel using vmap.

    On GPU, this can be 10-100x faster than sequential!

    Args:
        system: JaxSystemInterface
        t_span: (t0, t1)
        y0_batch: Array of initial conditions [batch_size, ...]
        params_batch: Array of parameters [batch_size, ...]
        saveat: Save times

    Returns:
        Batched solutions
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX required")

    sim = JaxSimulation(system)

    # Define single simulation
    def single_sim(y0, params):
        return sim.solve(t_span, y0, params, saveat=saveat)

    # Vectorize over batch dimension
    batch_solve = vmap(single_sim)

    # Run all simulations in parallel
    solutions = batch_solve(y0_batch, params_batch)

    return solutions


# ==============================================================================
# Demo and Benchmarks
# ==============================================================================

def demo_basic_usage():
    """Demonstrate basic JAX system usage"""
    if not JAX_AVAILABLE or not DIFFRAX_AVAILABLE:
        print("Skipping demo: JAX/Diffrax not available")
        return

    print("="*60)
    print("Demo 1: Basic JAX System")
    print("="*60)

    # Create system
    system = VanDerPolJax()

    # Initial conditions and parameters
    y0 = {'x': 0.0, 'y': 1.0}
    params = {'mu': 1.0}

    # Create simulation
    sim = JaxSimulation(system, solver='Dopri5')

    # Solve
    import time
    print("Solving Van der Pol (0 to 10s)...")
    t0 = time.perf_counter()
    solution = sim.solve(
        t_span=(0.0, 10.0),
        y0=y0,
        params=params,
        saveat=jnp.linspace(0, 10, 101)
    )
    t1 = time.perf_counter()

    print(f"Solved in {(t1-t0)*1000:.2f}ms")
    print(f"Final state: x={solution.ys['x'][-1]:.4f}, y={solution.ys['y'][-1]:.4f}")


def demo_composite_system():
    """Demonstrate coupled systems"""
    if not JAX_AVAILABLE or not DIFFRAX_AVAILABLE:
        print("Skipping demo: JAX/Diffrax not available")
        return

    print("\n" + "="*60)
    print("Demo 2: Composite System (Coupled Oscillators)")
    print("="*60)

    system = CoupledOscillatorsJax()

    # Nested initial conditions
    y0 = {
        'osc1': {'x': 0.0, 'y': 1.0},
        'osc2': {'x': 0.0, 'y': 0.0}
    }

    # Nested parameters
    params = {
        'osc1': {'mu': 1.0},
        'osc2': {'mu': 1.0},
        'coupling': 0.5  # osc1 influences osc2
    }

    sim = JaxSimulation(system)

    print("Solving coupled oscillators...")
    solution = sim.solve(
        t_span=(0.0, 10.0),
        y0=y0,
        params=params,
        saveat=jnp.linspace(0, 10, 101)
    )

    print(f"Final state:")
    print(f"  Osc1: x={solution.ys['osc1']['x'][-1]:.4f}")
    print(f"  Osc2: x={solution.ys['osc2']['x'][-1]:.4f}")


def demo_array_system():
    """Demonstrate array-based system"""
    if not JAX_AVAILABLE or not DIFFRAX_AVAILABLE:
        print("Skipping demo: JAX/Diffrax not available")
        return

    print("\n" + "="*60)
    print("Demo 3: Array System (10 Coupled Oscillators)")
    print("="*60)

    system = OscillatorArrayJax(n_oscillators=10)

    # Array-based initial conditions
    y0 = {
        'x': jnp.zeros(10),
        'v': jnp.ones(10)
    }

    params = {'k': 1.0, 'c': 0.1}

    sim = JaxSimulation(system)

    print("Solving 10 coupled oscillators...")
    import time
    t0 = time.perf_counter()
    solution = sim.solve(
        t_span=(0.0, 10.0),
        y0=y0,
        params=params,
        saveat=jnp.linspace(0, 10, 101)
    )
    t1 = time.perf_counter()

    print(f"Solved in {(t1-t0)*1000:.2f}ms")
    print(f"Final positions: {solution.ys['x'][-1]}")


def demo_autodiff():
    """Demonstrate automatic differentiation"""
    if not JAX_AVAILABLE or not DIFFRAX_AVAILABLE:
        print("Skipping demo: JAX/Diffrax not available")
        return

    print("\n" + "="*60)
    print("Demo 4: Automatic Differentiation")
    print("="*60)

    system = VanDerPolJax()

    # Initial conditions
    y0 = {'x': 0.0, 'y': 1.0}
    params = {'mu': 1.0}

    print("Computing sensitivity of final state to parameter 'mu'...")

    # This would require implementing parameter_sensitivity properly
    # (simplified here for prototype)
    print("(Gradient computation not fully implemented in prototype)")
    print("In full implementation: d(final_x)/d(mu) = ...")


def comparison_table():
    """Print comparison of scipy vs Numba vs JAX"""
    print("\n" + "="*60)
    print("Comparison: scipy vs Numba vs JAX")
    print("="*60)

    comparison = """
| Feature              | scipy (current) | + Numba      | + JAX        |
|----------------------|-----------------|--------------|--------------|
| CPU speedup          | 1x (baseline)   | 2-20x        | 2-10x        |
| GPU support          | ❌ No           | ❌ No        | ✅ Yes       |
| GPU speedup          | N/A             | N/A          | 10-100x      |
| Autodiff             | ❌ No           | ❌ No        | ✅ Yes       |
| API changes          | -               | Minimal      | Significant  |
| Learning curve       | -               | Low          | High         |
| Implementation time  | -               | 2 weeks      | 3 weeks      |
| Best for             | General use     | CPU speedup  | GPU/ML/optim |
"""
    print(comparison)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("="*60)
    print("Dynode + JAX: Functional Prototype")
    print("="*60)

    if not JAX_AVAILABLE:
        print("\n⚠️  JAX not installed!")
        print("Install with: pip install jax jaxlib")
        print("For GPU: pip install jax[cuda12]")

    if not DIFFRAX_AVAILABLE:
        print("\n⚠️  Diffrax not installed!")
        print("Install with: pip install diffrax")

    if JAX_AVAILABLE and DIFFRAX_AVAILABLE:
        # Run demos
        demo_basic_usage()
        demo_composite_system()
        demo_array_system()
        demo_autodiff()

    # Always show comparison
    comparison_table()

    print("\n" + "="*60)
    print("JAX Advantages:")
    print("  ✅ GPU/TPU acceleration (10-100x for large systems)")
    print("  ✅ Automatic differentiation (enables optimization)")
    print("  ✅ Vectorization (batch simulations in parallel)")
    print("  ✅ Composable transformations (jit, grad, vmap)")
    print("\nJAX Challenges:")
    print("  ⚠️  Functional paradigm (learning curve)")
    print("  ⚠️  Significant API redesign")
    print("  ⚠️  Large dependency (~500MB with CUDA)")
    print("  ⚠️  Longer compilation times (2-30s)")
    print("="*60)

    print("\n" + "="*60)
    print("Recommendation:")
    print("  1. Start with Numba for quick CPU wins")
    print("  2. Add JAX module for power users")
    print("  3. Let users choose based on needs")
    print("="*60)
