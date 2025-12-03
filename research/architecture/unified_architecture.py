"""
Unified Architecture: Combining Generator, Events, and Declarative Recording

This architecture resolves the fundamental tension between:
- Progressive stepping (dynode's current pattern)
- Batch solving (solve_ivp and modern solvers)

Key innovations:
1. Generator-based iteration (familiar, flexible, supports early termination)
2. Declarative recording (no callbacks, efficient batch processing)
3. Event-based termination (leverage solve_ivp's native events)
4. Backward compatible API (can wrap old simulate() pattern)
5. Configurable batching (performance optimization)
"""

from scipy.integrate import solve_ivp
import numpy as np
from typing import Generator, Tuple, Callable, Optional, List, Dict
from dataclasses import dataclass, field


# ============================================================================
# DECLARATIVE RECORDING
# ============================================================================

@dataclass
class RecorderConfig:
    """Configuration for what to record during simulation."""

    variables: Dict[str, Callable] = field(default_factory=dict)

    def add_variable(self, name: str, extractor: Callable):
        """Add a variable to record.

        Args:
            name: Variable name
            extractor: Function that extracts value from current state
                      Called as: extractor() -> float or np.ndarray
        """
        self.variables[name] = extractor
        return self

    def extract_all(self) -> Dict[str, float]:
        """Extract all variables at current simulation state."""
        return {name: extractor() for name, extractor in self.variables.items()}


class RecordedData:
    """Container for recorded simulation data."""

    def __init__(self):
        self.time = []
        self._data = {}

    def add_sample(self, t: float, values: Dict[str, float]):
        """Add a time sample."""
        self.time.append(t)
        for name, value in values.items():
            if name not in self._data:
                self._data[name] = []
            self._data[name].append(value)

    def __getattr__(self, name):
        """Access recorded data as attributes."""
        if name.startswith('_') or name == 'time':
            return object.__getattribute__(self, name)
        return np.array(self._data.get(name, []))

    def __repr__(self):
        vars_str = ', '.join(self._data.keys())
        return f"RecordedData({len(self.time)} samples: {vars_str})"


# ============================================================================
# EVENT-BASED TERMINATION
# ============================================================================

class TerminationEvent:
    """Declarative termination condition that becomes a solve_ivp event."""

    def __init__(self,
                 condition: Callable[[], float],
                 direction: int = 0,
                 terminal: bool = True):
        """
        Args:
            condition: Function returning value to monitor (zero-crossing detected)
            direction: -1 (decreasing), 0 (both), +1 (increasing)
            terminal: If True, stops integration when triggered
        """
        self.condition = condition
        self.direction = direction
        self.terminal = terminal

    def to_event_function(self, dispatch_func):
        """Convert to solve_ivp event function."""
        def event_func(t, y):
            # Must dispatch states before calling condition
            dispatch_func(y)
            return self.condition()

        event_func.terminal = self.terminal
        event_func.direction = self.direction
        return event_func


# ============================================================================
# UNIFIED SIMULATION
# ============================================================================

class UnifiedSimulation:
    """
    Modern simulation architecture compatible with solve_ivp and batch solvers.

    Provides three usage patterns:
    1. Generator iteration (progressive with batching)
    2. Declarative recording (efficient batch processing)
    3. Legacy observer API (backward compatibility)
    """

    def __init__(self):
        self._systems = []
        self._t = 0

        # New pattern: declarative
        self._recorder = RecorderConfig()
        self._events = []

        # Legacy pattern: observers
        self._observers = []

    def add_system(self, system):
        """Add a system to the simulation."""
        self._systems.append(system)

    # ------------------------------------------------------------------------
    # PATTERN 1: Generator-based iteration
    # ------------------------------------------------------------------------

    def run(self,
            t_end: float,
            dt: float,
            method: str = 'RK45',
            batch_size: int = 10,
            **solver_kwargs) -> Generator[Tuple[float, np.ndarray], Optional[bool], None]:
        """
        Generator that yields (time, state) at regular intervals.

        This is the RECOMMENDED modern pattern. Provides:
        - Familiar iteration interface
        - Early termination support
        - Batching optimization under the hood
        - Full control over simulation flow

        Args:
            t_end: End time
            dt: Yield interval
            method: Integration method ('RK45', 'DOP853', 'LSODA', etc.)
            batch_size: Number of steps to solve at once (performance tuning)
            **solver_kwargs: Additional arguments for solve_ivp

        Yields:
            (t, y) tuples

        Receives:
            Optional bool: False to stop integration

        Example:
            >>> for t, y in sim.run(t_end=10, dt=0.1):
            ...     print(f"t={t}, y={y}")
            ...     if some_condition:
            ...         break  # Stop integration
        """
        t_current = self._t
        y_current = self._collect_states()

        # Build derivative function
        func = self._make_derivative_function()

        # Convert events to solve_ivp format
        events = [evt.to_event_function(self._dispatch_states) for evt in self._events]

        # Yield initial state
        should_continue = yield (t_current, y_current)
        if should_continue is False:
            self._t = t_current
            return

        # Integration loop with batching
        while t_current < t_end:
            # Determine batch window
            t_batch_end = min(t_current + batch_size * dt, t_end)
            num_steps = int((t_batch_end - t_current) / dt)

            if num_steps == 0:
                break

            t_eval = t_current + np.arange(1, num_steps + 1) * dt

            # Solve batch
            result = solve_ivp(
                func,
                t_span=(t_current, t_batch_end),
                y0=y_current,
                method=method,
                t_eval=t_eval,
                events=events,
                **solver_kwargs
            )

            if not result.success:
                raise RuntimeError(f"Solver failed: {result.message}")

            # Yield each point in batch
            for i, (t, y) in enumerate(zip(result.t, result.y.T)):
                self._dispatch_states(y)
                should_continue = yield (t, y)

                if should_continue is False:
                    self._t = t
                    return

            # Check for event termination
            if result.status == 1:  # Event triggered
                break

            # Update for next batch
            t_current = result.t[-1]
            y_current = result.y[:, -1]

        # Update final state
        self._t = t_current
        self._dispatch_states(y_current)

    # ------------------------------------------------------------------------
    # PATTERN 2: Declarative recording (most efficient)
    # ------------------------------------------------------------------------

    def record(self, name: str, extractor: Callable):
        """Declare a variable to record (declarative, no callbacks).

        Args:
            name: Variable name
            extractor: Function that extracts value from current state

        Returns:
            Self for chaining

        Example:
            >>> sim.record('position', lambda: sys.states.x)
            >>> sim.record('velocity', lambda: sys.states.v)
            >>> data = sim.simulate_batch(t_end=10, dt=0.1)
            >>> plt.plot(data.time, data.position)
        """
        self._recorder.add_variable(name, extractor)
        return self

    def add_termination_event(self,
                             condition: Callable[[], float],
                             direction: int = 0) -> 'UnifiedSimulation':
        """Add a termination condition (declarative, uses solve_ivp events).

        Args:
            condition: Function returning value to monitor (stops at zero-crossing)
            direction: -1 (decreasing), 0 (both), +1 (increasing)

        Returns:
            Self for chaining

        Example:
            >>> # Stop when position crosses zero (from positive to negative)
            >>> sim.add_termination_event(lambda: sys.states.x, direction=-1)
        """
        self._events.append(TerminationEvent(condition, direction, terminal=True))
        return self

    def simulate_batch(self,
                      t_end: float,
                      dt: float,
                      method: str = 'RK45',
                      **solver_kwargs) -> RecordedData:
        """
        Batch simulation with declarative recording.

        This is the MOST EFFICIENT pattern for recording data.
        Uses solve_ivp natively without progressive stepping overhead.

        Args:
            t_end: End time
            dt: Recording interval
            method: Integration method
            **solver_kwargs: Additional arguments for solve_ivp

        Returns:
            RecordedData with time and all declared variables

        Example:
            >>> sim.record('x', lambda: sys.states.x)
            >>> sim.record('v', lambda: sys.states.v)
            >>> data = sim.simulate_batch(t_end=10, dt=0.1)
            >>> plt.plot(data.time, data.x)
        """
        t_current = self._t
        y_current = self._collect_states()

        # Build derivative function
        func = self._make_derivative_function()

        # Convert events
        events = [evt.to_event_function(self._dispatch_states) for evt in self._events]

        # Evaluation points
        t_eval = t_current + np.arange(0, int((t_end - t_current) / dt) + 1) * dt
        t_eval = t_eval[t_eval <= t_end]

        # Single batch solve
        result = solve_ivp(
            func,
            t_span=(t_current, t_end),
            y0=y_current,
            method=method,
            t_eval=t_eval,
            events=events,
            **solver_kwargs
        )

        if not result.success:
            raise RuntimeError(f"Solver failed: {result.message}")

        # Extract recorded variables
        recorded = RecordedData()
        for t, y in zip(result.t, result.y.T):
            self._dispatch_states(y)
            values = self._recorder.extract_all()
            recorded.add_sample(t, values)

        # Update final state
        self._t = result.t[-1]
        self._dispatch_states(result.y[:, -1])

        return recorded

    # ------------------------------------------------------------------------
    # PATTERN 3: Legacy observer API (backward compatibility)
    # ------------------------------------------------------------------------

    def add_observer(self, observer: Callable):
        """Add observer (legacy API, backward compatible).

        Note: This uses the generator pattern under the hood, so you still
        get batching optimization while maintaining the old API.
        """
        self._observers.append(observer)

    def simulate(self,
                t: float,
                observer_dt: float,
                fixed_step: bool = False,
                integrator: str = 'RK45',
                batch_size: int = 10,
                **kwargs) -> float:
        """
        Legacy simulate() API for backward compatibility.

        Wraps the generator pattern to maintain old behavior while
        using modern batch solving under the hood.

        Args:
            t: Total time to progress
            observer_dt: Timestep with which observers are invoked
            fixed_step: If True, use fixed step size
            integrator: Which solver to use (maps to solve_ivp method)
            batch_size: Performance tuning parameter
            **kwargs: Additional solver arguments

        Returns:
            Current simulation time
        """
        if not self._systems:
            raise RuntimeError("Need at least 1 system in the simulation!")

        # Map old integrator names to solve_ivp methods
        method_map = {
            'dopri5': 'RK45',
            'dop853': 'DOP853',
            'vode': 'LSODA',
            'lsoda': 'LSODA',
        }
        method = method_map.get(integrator.lower(), integrator)

        # Fixed step workaround
        if fixed_step:
            kwargs.update({'max_step': observer_dt})

        # Use generator pattern
        for time, state in self.run(self._t + t, observer_dt, method, batch_size, **kwargs):
            # Inform observers
            terminate = False
            for obs in self._observers:
                if obs(time, state):
                    terminate = True
                    break

            if terminate:
                break

        return self._t

    # ------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------

    def _collect_states(self):
        """Collect states from all systems."""
        return np.concatenate([sys.get_states() for sys in self._systems], axis=None)

    def _dispatch_states(self, states):
        """Dispatch states to all systems."""
        idx = 0
        for sys in self._systems:
            idx = sys.dispatch_states(idx, states)

    def _make_derivative_function(self):
        """Create derivative function for solver."""
        dy = np.zeros(len(self._collect_states()))

        def func(t, y):
            self._dispatch_states(y)
            for sys in self._systems:
                sys._step(t)

            idx = 0
            for sys in self._systems:
                idx = sys.get_ders(idx, dy)

            return dy

        return func


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_generator_pattern():
    """Example: Generator-based iteration (recommended for interactive use)"""
    print("=== PATTERN 1: Generator Iteration ===")

    class MockSystem:
        def __init__(self):
            self.state = np.array([1.0, 0.0])
        def get_states(self):
            return self.state
        def dispatch_states(self, idx, states):
            self.state[:] = states[idx:idx+2]
            return idx + 2
        def get_ders(self, idx, ders):
            ders[idx] = -self.state[0]
            ders[idx+1] = self.state[1]
            return idx + 2
        def _step(self, t):
            pass

    sys = MockSystem()
    sim = UnifiedSimulation()
    sim.add_system(sys)

    print("Running with early termination:")
    count = 0
    for t, y in sim.run(t_end=10.0, dt=0.2, batch_size=5):
        print(f"  t={t:.1f}, y={y}")
        count += 1
        if count >= 5:
            print("  Stopping early!")
            break

    print(f"Executed {count} steps (batched 5 at a time)")


def example_declarative_recording():
    """Example: Declarative recording (recommended for data collection)"""
    print("\n=== PATTERN 2: Declarative Recording ===")

    class OscillatorSystem:
        def __init__(self):
            self.state = np.array([1.0, 0.0])  # [position, velocity]
        def get_states(self):
            return self.state
        def dispatch_states(self, idx, states):
            self.state[:] = states[idx:idx+2]
            return idx + 2
        def get_ders(self, idx, ders):
            # Simple harmonic oscillator: x'' = -x
            ders[idx] = self.state[1]      # dx/dt = v
            ders[idx+1] = -self.state[0]   # dv/dt = -x
            return idx + 2
        def _step(self, t):
            pass

    sys = OscillatorSystem()
    sim = UnifiedSimulation()
    sim.add_system(sys)

    # Declaratively specify what to record
    sim.record('position', lambda: sys.state[0])
    sim.record('velocity', lambda: sys.state[1])
    sim.record('energy', lambda: 0.5 * (sys.state[0]**2 + sys.state[1]**2))

    # Single efficient batch solve
    data = sim.simulate_batch(t_end=10.0, dt=0.5)

    print(f"Recorded {len(data.time)} samples")
    print(f"Times: {data.time}")
    print(f"Positions: {data.position}")
    print(f"Energy (should be constant): {data.energy}")
    print(f"Data object: {data}")


def example_with_termination_event():
    """Example: Declarative termination event"""
    print("\n=== PATTERN 2b: Declarative Termination ===")

    class DecaySystem:
        def __init__(self):
            self.state = np.array([10.0])
        def get_states(self):
            return self.state
        def dispatch_states(self, idx, states):
            self.state[:] = states[idx:idx+1]
            return idx + 1
        def get_ders(self, idx, ders):
            ders[idx] = -2.0 * self.state[0]  # Exponential decay
            return idx + 1
        def _step(self, t):
            pass

    sys = DecaySystem()
    sim = UnifiedSimulation()
    sim.add_system(sys)

    # Record data
    sim.record('value', lambda: sys.state[0])

    # Stop when value crosses 1.0 from above
    threshold = 1.0
    sim.add_termination_event(lambda: sys.state[0] - threshold, direction=-1)

    # Run until event triggers
    data = sim.simulate_batch(t_end=100.0, dt=0.1)

    print(f"Stopped at t={data.time[-1]:.3f} when value={data.value[-1]:.3f}")
    print(f"Threshold was {threshold}")


def example_legacy_compatibility():
    """Example: Legacy observer API (backward compatible)"""
    print("\n=== PATTERN 3: Legacy Observer API ===")

    class SimpleSystem:
        def __init__(self):
            self.state = np.array([1.0])
        def get_states(self):
            return self.state
        def dispatch_states(self, idx, states):
            self.state[:] = states[idx:idx+1]
            return idx + 1
        def get_ders(self, idx, ders):
            ders[idx] = -self.state[0]
            return idx + 1
        def _step(self, t):
            pass

    sys = SimpleSystem()
    sim = UnifiedSimulation()
    sim.add_system(sys)

    # Old-style observer
    times = []
    values = []

    def my_observer(t, y):
        times.append(t)
        values.append(y[0])
        if y[0] < 0.5:
            print(f"  Observer stopping at t={t:.2f}, y={y[0]:.3f}")
            return True  # Terminate
        return False

    sim.add_observer(my_observer)

    # Old API still works!
    final_t = sim.simulate(t=10.0, observer_dt=0.2)

    print(f"Simulated to t={final_t:.2f}")
    print(f"Recorded {len(times)} samples via observer")


if __name__ == "__main__":
    example_generator_pattern()
    example_declarative_recording()
    example_with_termination_event()
    example_legacy_compatibility()

    print("\n" + "="*70)
    print("ARCHITECTURE SUMMARY")
    print("="*70)
    print("\nThree usage patterns, all using solve_ivp under the hood:")
    print("\n1. Generator iteration (sim.run)")
    print("   - Familiar for-loop interface")
    print("   - Early termination support")
    print("   - Configurable batching")
    print("   - Best for: Interactive exploration, custom logic")

    print("\n2. Declarative recording (sim.simulate_batch)")
    print("   - Most efficient (single batch solve)")
    print("   - Clean separation of concerns")
    print("   - Event-based termination")
    print("   - Best for: Data collection, parameter sweeps")

    print("\n3. Legacy observer API (sim.simulate)")
    print("   - Backward compatible")
    print("   - Uses generator pattern internally")
    print("   - Best for: Existing code migration")

    print("\nKey benefits:")
    print("✓ Leverages solve_ivp's performance and features")
    print("✓ Native event detection support")
    print("✓ Flexible batching (user-controllable)")
    print("✓ Backward compatible with existing dynode code")
    print("✓ Clean, modern API for new projects")
