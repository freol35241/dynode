"""
Alternative Architecture 2: Generator-Based Simulation

Key idea: Yield control at regular intervals while batching under the hood
"""

from scipy.integrate import solve_ivp
import numpy as np
from typing import Generator, Tuple


class GeneratorSimulation:
    """
    Simulation that yields control at regular intervals.

    User gets illusion of progressive stepping, but we batch solve under the hood.
    """

    def __init__(self):
        self._systems = []
        self._t = 0

    def add_system(self, system):
        self._systems.append(system)

    def run(self,
            t_end: float,
            dt: float,
            method: str = 'RK45',
            batch_size: int = 10) -> Generator[Tuple[float, np.ndarray], bool, None]:
        """
        Generator that yields (time, state) at regular intervals.

        Args:
            t_end: End time
            dt: Yield interval
            method: Integration method
            batch_size: Number of steps to solve at once (optimization)

        Yields:
            (t, y) tuples

        Receives:
            should_continue (bool): False to stop integration

        Example:
            for t, y in sim.run(t_end=10, dt=0.1):
                print(f"t={t}, y={y}")
                if some_condition:
                    break  # Stop integration
        """
        t_current = self._t

        # Collect initial state
        y_current = self._collect_states()

        # Build derivative function
        def func(t, y):
            self._dispatch_states(y)
            for sys in self._systems:
                sys._step(t)
            dy = np.zeros_like(y)
            self._collect_ders(dy)
            return dy

        # Yield initial state
        should_continue = yield (t_current, y_current)
        if should_continue is False:
            self._t = t_current
            return

        # Integration loop with batching
        while t_current < t_end:
            # Determine batch window
            t_batch_end = min(t_current + batch_size * dt, t_end)

            # Create evaluation points for this batch
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
                t_eval=t_eval
            )

            if not result.success:
                raise RuntimeError(f"Solver failed: {result.message}")

            # Yield each point in batch
            for i, (t, y) in enumerate(zip(result.t, result.y.T)):
                self._dispatch_states(y)

                should_continue = yield (t, y)

                if should_continue is False:
                    # Early termination
                    self._t = t
                    return

            # Update for next batch
            t_current = result.t[-1]
            y_current = result.y[:, -1]

        # Update final state
        self._t = t_current
        self._dispatch_states(y_current)

    def _collect_states(self):
        return np.concatenate([sys.get_states() for sys in self._systems], axis=None)

    def _dispatch_states(self, states):
        idx = 0
        for sys in self._systems:
            idx = sys.dispatch_states(idx, states)

    def _collect_ders(self, ders):
        idx = 0
        for sys in self._systems:
            idx = sys.get_ders(idx, ders)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_basic():
    """Basic usage - same feel as current dynode"""
    print("=== Basic Generator Usage ===")

    # Mock system for demo
    class MockSystem:
        def __init__(self):
            self.state = np.array([1.0, 0.0])
        def get_states(self):
            return self.state
        def dispatch_states(self, idx, states):
            self.state[:] = states[idx:idx+2]
            return idx + 2
        def get_ders(self, idx, ders):
            ders[idx:idx+2] = [-self.state[0], self.state[1]]
            return idx + 2
        def _step(self, t):
            pass

    sys = MockSystem()
    sim = GeneratorSimulation()
    sim.add_system(sys)

    # Use like an iterator
    count = 0
    for t, y in sim.run(t_end=1.0, dt=0.1):
        count += 1
        print(f"  t={t:.1f}, y={y}")
        if count >= 5:
            print("  Stopping early!")
            break

    print(f"  Executed {count} steps")


def example_with_recorder():
    """Recording data from generator"""
    print("\n=== Generator with Recording ===")

    class MockSystem:
        def __init__(self):
            self.state = np.array([1.0])
        def get_states(self):
            return self.state
        def dispatch_states(self, idx, states):
            self.state[:] = states[idx:idx+1]
            return idx + 1
        def get_ders(self, idx, ders):
            ders[idx:idx+1] = [-self.state[0]]
            return idx + 1
        def _step(self, t):
            pass

    sys = MockSystem()
    sim = GeneratorSimulation()
    sim.add_system(sys)

    # Manual recording
    times = []
    values = []

    for t, y in sim.run(t_end=2.0, dt=0.2):
        times.append(t)
        values.append(y[0])

    print(f"  Recorded {len(times)} points")
    print(f"  Times: {times}")
    print(f"  Values: {values}")


def example_conditional_termination():
    """Termination based on condition"""
    print("\n=== Conditional Termination ===")

    class MockSystem:
        def __init__(self):
            self.state = np.array([10.0])
        def get_states(self):
            return self.state
        def dispatch_states(self, idx, states):
            self.state[:] = states[idx:idx+1]
            return idx + 1
        def get_ders(self, idx, ders):
            ders[idx:idx+1] = [-2.0 * self.state[0]]
            return idx + 1
        def _step(self, t):
            pass

    sys = MockSystem()
    sim = GeneratorSimulation()
    sim.add_system(sys)

    # Stop when state drops below threshold
    threshold = 1.0
    for t, y in sim.run(t_end=10.0, dt=0.1):
        print(f"  t={t:.2f}, y={y[0]:.3f}")
        if y[0] < threshold:
            print(f"  Threshold {threshold} crossed!")
            break


if __name__ == "__main__":
    example_basic()
    example_with_recorder()
    example_conditional_termination()

    print("\n=== Key Benefits ===")
    print("✓ Familiar iterator interface")
    print("✓ Early termination support")
    print("✓ Batching under the hood (configurable)")
    print("✓ Backward compatible (can wrap in old API)")
