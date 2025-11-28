# Dynode Acceleration Investigation Summary

**Date:** 2025-11-28
**Investigations Completed:** JAX Integration, Numba Integration

---

## Executive Summary

Both **JAX** and **Numba** can significantly accelerate dynode, but they serve different use cases:

| Criterion | Numba | JAX |
|-----------|-------|-----|
| **Best for** | CPU-only acceleration | GPU + autodiff + optimization |
| **Implementation effort** | 2 weeks | 2.5-3 weeks |
| **API impact** | Minimal (decorators) | Significant (functional redesign) |
| **Speedup (CPU)** | 2-20x | 2-10x |
| **Speedup (GPU)** | N/A | 10-100x |
| **Learning curve** | Low | High |
| **Autodiff** | ❌ No | ✅ Yes |
| **Breaking changes** | ✅ None | ❌ Yes (new module) |

---

## Recommendations by Use Case

### Recommendation 1: **Start with Numba** (Quick Win)
**Timeline:** 2 weeks
**Rationale:**
- ✅ Low-hanging fruit: 5-20x CPU speedup with minimal changes
- ✅ Works with existing API (backward compatible)
- ✅ Immediate value for current users
- ✅ Validates performance improvement strategy

**Implementation:** Lightweight JIT decoration pattern
```python
from numba import njit

class VanDerPol(SystemInterface):
    @staticmethod
    @njit
    def _compute(x, y, mu):
        return y, mu * (1 - x**2) * y - x

    def do_step(self, time):
        self.ders.dx, self.ders.dy = self._compute(
            self.states.x, self.states.y, self.inputs.mu
        )
```

**Impact on System developers:**
- 5-10 minutes to convert existing systems
- Opt-in (no forced migration)
- Same API, same mental model

---

### Recommendation 2: **Add JAX Module** (Advanced Users)
**Timeline:** 2.5-3 weeks (after Numba)
**Rationale:**
- ✅ Enables GPU acceleration for large systems
- ✅ Automatic differentiation for parameter optimization
- ✅ Positions dynode for ML/scientific computing use cases
- ✅ Separate module = no disruption to existing users

**Implementation:** New `dynode.jax` submodule
```python
from dynode.jax import JaxSystemInterface, JaxSimulation

class VanDerPol(JaxSystemInterface):
    def dynamics(self, t, state, params):
        return {'x': state['y'], 'y': params['mu'] * (1 - state['x']**2) * state['y'] - state['x']}

sim = JaxSimulation(system=VanDerPol(), solver='Dopri5')
result = sim.solve(t_span=(0, 100), y0={'x': 0.0, 'y': 1.0}, params={'mu': 1.0})
```

**Impact on System developers:**
- Functional programming paradigm (higher learning curve)
- Separate API (co-exists with classic dynode)
- Opt-in for power users

---

## Decision Matrix

### Scenario 1: Small Systems (<10 states), CPU-only
**Recommendation:** **Numba**
- JAX compilation overhead not worth it
- Numba 2-5x speedup sufficient
- No API changes needed

### Scenario 2: Large Systems (>50 states), GPU available
**Recommendation:** **JAX**
- 10-100x speedup on GPU
- Worth the functional API redesign
- Consider batch simulations (vmap)

### Scenario 3: Parameter Optimization / Sensitivity Analysis
**Recommendation:** **JAX**
- Automatic differentiation critical
- Gradient-based optimization 10-50x faster
- GPU batching for Monte Carlo

### Scenario 4: Existing Production Code
**Recommendation:** **Numba first, JAX optional**
- Minimize migration effort
- Incremental performance gains
- Add JAX later if GPU needs arise

---

## Implementation Roadmap

### Phase 1: Numba Integration (2 weeks)
**Week 1:**
- [ ] Create `dynode/numba_utils.py` helper module
- [ ] Convert 3 example systems (VanDerPol, SingleDegreeMass, composite)
- [ ] Benchmark vs plain Python (target >3x speedup)
- [ ] Write user tutorial

**Week 2:**
- [ ] Add optional dependency (`pip install dynode[numba]`)
- [ ] CI testing for Numba compatibility
- [ ] Performance documentation
- [ ] Troubleshooting guide

**Deliverable:** `dynode` v0.5.0 with optional Numba acceleration

---

### Phase 2: JAX Module (3 weeks, optional)
**Week 1:**
- [ ] Create `dynode/jax/` submodule structure
- [ ] Implement `JaxSystemInterface` (functional base)
- [ ] Integrate Diffrax for ODE solving
- [ ] Test single VanDerPol system

**Week 2:**
- [ ] Design subsystem composition pattern
- [ ] Implement connection mechanism
- [ ] Add transformation utilities (jit, vmap, grad)
- [ ] GPU testing (if available)

**Week 3:**
- [ ] Documentation: migration guide
- [ ] Examples: GPU acceleration, parameter optimization
- [ ] Performance benchmarks
- [ ] Release `dynode` v0.6.0 with JAX module

**Deliverable:** `dynode.jax` for advanced users

---

## Prototype Examples

### Example 1: Numba Pattern (Minimal Changes)

**Before:**
```python
class PowerGridBus(SystemInterface):
    def __init__(self):
        super().__init__()
        self.states.delta = 0.0
        self.states.omega = 0.0
        self.ders.d_delta = 0.0
        self.ders.d_omega = 0.0
        self.inputs.P_mech = 0.0
        self.inputs.P_elec = 0.0
        self.inputs.D = 0.1
        self.inputs.M = 10.0

    def do_step(self, time):
        self.ders.d_delta = self.states.omega
        self.ders.d_omega = (self.inputs.P_mech - self.inputs.P_elec -
                             self.inputs.D * self.states.omega) / self.inputs.M
```

**After (with Numba):**
```python
from numba import njit

class PowerGridBus(SystemInterface):
    def __init__(self):
        # ... same initialization ...

    @staticmethod
    @njit
    def _compute(delta, omega, P_mech, P_elec, D, M):
        d_delta = omega
        d_omega = (P_mech - P_elec - D * omega) / M
        return d_delta, d_omega

    def do_step(self, time):
        self.ders.d_delta, self.ders.d_omega = self._compute(
            self.states.delta, self.states.omega,
            self.inputs.P_mech, self.inputs.P_elec,
            self.inputs.D, self.inputs.M
        )
```

**Expected speedup:** 5-15x for 100-bus grid

---

### Example 2: JAX Pattern (Functional)

```python
from dynode.jax import JaxSystemInterface, JaxSimulation
import jax.numpy as jnp

class PowerGridBus(JaxSystemInterface):
    def dynamics(self, t, state, params):
        delta, omega = state['delta'], state['omega']
        P_mech, P_elec = params['P_mech'], params['P_elec']
        D, M = params['D'], params['M']

        return {
            'delta': omega,
            'omega': (P_mech - P_elec - D * omega) / M
        }

# Simulate 100 buses in parallel on GPU
from jax import vmap

def simulate_grid(initial_conditions, parameters):
    """Vectorized over all buses"""
    sim = JaxSimulation(system=PowerGridBus(), solver='Dopri5')
    return sim.solve(t_span=(0, 10), y0=initial_conditions, params=parameters)

# Batch over 100 buses
results = vmap(simulate_grid)(initial_conditions_array, parameters_array)
```

**Expected speedup:** 50-100x vs sequential scipy (100 buses on GPU)

---

## Cost-Benefit Analysis

### Numba
**Cost:**
- 2 weeks development
- User education (Numba patterns)
- Optional dependency management

**Benefit:**
- 5-20x CPU speedup (amortized)
- Minimal user migration
- Positions dynode as "fast by default"
- **ROI: High** (low cost, immediate value)

### JAX
**Cost:**
- 3 weeks development
- Significant API redesign
- User learning curve (functional programming)
- GPU infrastructure requirements

**Benefit:**
- 10-100x GPU speedup
- Automatic differentiation (new capabilities)
- Modern ML/scientific computing workflows
- **ROI: Medium** (higher cost, niche use cases)

---

## Risks & Mitigations

### Numba Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Variable speedups | Users disappointed | Clear documentation of when Numba helps |
| Compilation overhead | Perceived slowness | Document warmup, use caching |
| Type errors | Debugging difficulty | Provide error guide, test templates |

### JAX Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Functional API confusion | Low adoption | Extensive examples, migration guide |
| GPU availability | Feature unused | CPU fallback, document performance |
| Breaking changes | User frustration | Separate module, maintain classic API |

---

## Conclusion

**Recommended Strategy:**

1. **Implement Numba first** (2 weeks)
   - Quick win, low risk
   - Validates acceleration approach
   - Broad applicability

2. **Evaluate need for JAX** (based on user feedback)
   - If GPU demand is high → implement JAX module
   - If autodiff is requested → JAX becomes critical
   - Otherwise → Numba may be sufficient

3. **Long-term: Dual support**
   - Numba as default recommendation (CPU)
   - JAX as advanced option (GPU/autodiff)
   - Let users choose based on needs

**Next Steps:**
1. Create Numba prototype (1 week)
2. Benchmark representative systems
3. Decision gate: proceed if >3x average speedup
4. Full implementation + documentation

---

## Performance Expectations

### Conservative Estimates (Numba)
- Small systems (2-5 states): 1.5-3x
- Medium systems (10-20 states): 3-8x
- Large systems (50+ states): 10-20x
- **Average across use cases: 4-6x**

### Optimistic Estimates (JAX on GPU)
- Medium systems: 10-30x
- Large systems: 30-100x
- Batch simulations: 50-200x
- **Average for GPU-suitable cases: 20-50x**

---

**Author:** Claude (dynode investigation)
**Repository:** https://github.com/freol35241/dynode
**Branch:** `claude/dynode-jax-investigation-01VRW5pTVdHZ4Q9VmP7VEdz7`
