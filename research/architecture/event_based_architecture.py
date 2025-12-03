"""
Alternative Architecture 1: Event-Based Simulation

Key idea: Separate passive observation (recording) from active control (events)
"""

from scipy.integrate import solve_ivp
import numpy as np
from typing import List, Callable, Optional


class Recorder:
    """Passive data collector - doesn't affect integration"""

    def __init__(self):
        self.data = {}
        self.times = []

    def declare_variable(self, name: str, extractor: Callable):
        """Declare what to record, not when"""
        self.data[name] = {
            'extractor': extractor,
            'values': []
        }

    def fill_from_result(self, result, systems):
        """Called AFTER integration - extract recorded variables"""
        self.times = result.t.tolist()

        for name, spec in self.data.items():
            spec['values'] = []
            for i, (t, y) in enumerate(zip(result.t, result.y.T)):
                # Dispatch states to systems so extractors can access them
                dispatch_states(y, systems)
                value = spec['extractor']()
                spec['values'].append(value)

    def __getitem__(self, name):
        """Access recorded data"""
        return np.array(self.data[name]['values'])


class TerminationCondition:
    """Active control - becomes an event in solve_ivp"""

    def __init__(self, condition: Callable, direction=0):
        """
        Args:
            condition: Function(systems) -> float, triggers when crosses zero
            direction: -1 (neg to pos), +1 (pos to neg), 0 (any)
        """
        self.condition = condition
        self.direction = direction

    def to_event(self, systems):
        """Convert to solve_ivp event function"""
        def event(t, y):
            dispatch_states(y, systems)
            return self.condition()

        event.terminal = True
        event.direction = self.direction
        return event


class EventBasedSimulation:
    """
    Simulation using solve_ivp's native features:
    - t_eval for recording points
    - events for termination/detection
    - dense_output for interpolation
    """

    def __init__(self):
        self._systems = []
        self._recorders = []
        self._termination_conditions = []
        self._event_detectors = []
        self._t = 0

    def add_system(self, system):
        self._systems.append(system)

    def add_recorder(self, recorder: Recorder):
        """Add passive recorder"""
        self._recorders.append(recorder)

    def add_termination_condition(self, condition: TerminationCondition):
        """Add condition that stops integration"""
        self._termination_conditions.append(condition)

    def add_event_detector(self, event: TerminationCondition):
        """Add non-terminal event detector"""
        self._event_detectors.append(event)

    def simulate(self,
                 t: float,
                 recording_dt: Optional[float] = None,
                 method: str = 'RK45',
                 dense_output: bool = False,
                 **solver_options) -> 'SimulationResult':
        """
        Simulate forward in time using batch solving.

        Args:
            t: Total time to integrate
            recording_dt: Interval for recording (None = solver-determined)
            method: Integration method
            dense_output: Enable continuous solution

        Returns:
            SimulationResult with .sol, .t_events, etc.
        """
        # Collect initial state
        y0 = collect_states(self._systems)

        # Build derivative function
        dy = np.zeros_like(y0)
        def func(t, y):
            dispatch_states(y, self._systems)
            for sys in self._systems:
                sys._step(t)
            collect_ders(dy, self._systems)
            return dy

        # Build t_eval for recorders
        if recording_dt:
            num_points = int(t / recording_dt) + 1
            t_eval = self._t + np.linspace(0, t, num_points)
        else:
            t_eval = None

        # Build events
        events = []
        for cond in self._termination_conditions:
            events.append(cond.to_event(self._systems))
        for evt in self._event_detectors:
            evt_func = evt.to_event(self._systems)
            evt_func.terminal = False
            events.append(evt_func)

        # Solve!
        result = solve_ivp(
            func,
            t_span=(self._t, self._t + t),
            y0=y0,
            method=method,
            t_eval=t_eval,
            events=events if events else None,
            dense_output=dense_output,
            **solver_options
        )

        # Fill recorders
        for recorder in self._recorders:
            recorder.fill_from_result(result, self._systems)

        # Update simulation state
        self._t = result.t[-1]
        dispatch_states(result.y[:, -1], self._systems)

        return result


# Helper functions (would be in dynode.simulation)
def collect_states(systems):
    return np.concatenate([sys.get_states() for sys in systems], axis=None)

def dispatch_states(states, systems):
    idx = 0
    for sys in systems:
        idx = sys.dispatch_states(idx, states)

def collect_ders(ders, systems):
    idx = 0
    for sys in systems:
        idx = sys.get_ders(idx, ders)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from dynode import SystemInterface

    class VanDerPol(SystemInterface):
        def __init__(self):
            super().__init__()
            self.inputs.mu = 1.0
            self.states.x = 1.0
            self.ders.dx = 0.0
            self.states.y = 0.0
            self.ders.dy = 0.0

        def do_step(self, time):
            mu = self.inputs.mu
            x = self.states.x
            y = self.states.y
            self.ders.dx = y
            self.ders.dy = mu * (1 - x**2) * y - x

    # Create system
    sys = VanDerPol()

    # Create recorder (declarative!)
    rec = Recorder()
    rec.declare_variable('x', lambda: sys.states.x)
    rec.declare_variable('y', lambda: sys.states.y)

    # Create termination condition
    def stop_when_x_negative():
        return sys.states.x  # Crosses zero when x goes negative

    terminator = TerminationCondition(stop_when_x_negative, direction=-1)

    # Create simulation
    sim = EventBasedSimulation()
    sim.add_system(sys)
    sim.add_recorder(rec)
    # sim.add_termination_condition(terminator)  # Uncomment to test early termination

    # Run simulation
    result = sim.simulate(t=10.0, recording_dt=0.1, method='RK45')

    print(f"Integration successful: {result.success}")
    print(f"Final time: {result.t[-1]:.2f}")
    print(f"Recorded {len(rec.times)} points")
    print(f"x range: [{rec['x'].min():.3f}, {rec['x'].max():.3f}]")

    if result.t_events and any(len(evt) > 0 for evt in result.t_events):
        print(f"Event detected at: {result.t_events[0]}")
