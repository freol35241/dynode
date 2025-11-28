# Numba Integration: Required Core Changes

## Summary: **One New File, Zero Core Changes**

For the lightweight Numba integration (Approach 1), dynode's core code requires **no modifications**. The entire integration consists of:

1. ✅ Adding optional helper utilities (`dynode/numba_utils.py`)
2. ✅ Updating documentation
3. ✅ Adding examples and tests

---

## Files That Need Changes: **1 File**

### 1. NEW: `dynode/numba_utils.py` (Optional Helpers)

**Purpose:** Make Numba integration easier for users

**Contents:**
- `@numba_rhs` decorator (auto-extract states/inputs pattern)
- `@jit_system_method` decorator (simple wrapper)
- Re-export `njit` for convenience

**Size:** ~100 lines

**Note:** This is **purely optional**. Users can use Numba without this by:
```python
from numba import njit

class MySystem(SystemInterface):
    @staticmethod
    @njit
    def _compute(x, y):
        return x + y, x - y

    def do_step(self, time):
        self.ders.dx, self.ders.dy = self._compute(
            self.states.x, self.states.y
        )
```

---

## Files That DON'T Need Changes

### ❌ `dynode/system.py` - No changes
**Why:**
- `SystemInterface` is agnostic to how `do_step()` is implemented
- Users can freely use `@njit` decorated methods
- Connections/subsystems work in Python (don't need JIT)

**Proof:**
```python
# Current code works fine with Numba
class VanDerPol(SystemInterface):  # Unchanged base class
    @staticmethod
    @njit  # User adds this
    def _compute(x, y, mu):
        return y, mu * (1 - x**2) * y - x

    def do_step(self, time):  # Unchanged signature
        self.ders.dx, self.ders.dy = self._compute(...)
```

---

### ❌ `dynode/simulation.py` - No changes
**Why:**
- `scipy.integrate.ode` already accepts Numba-JIT'd functions
- The `func(t, y)` passed to solver can be compiled or not - solver doesn't care

**Current code (simulation.py:152-159):**
```python
def func(t, y):
    dispatch_states(y, self.systems)
    for sys in self.systems:
        sys._step(t)  # Calls do_step() - can be JIT compiled internally
    collect_ders(dy, self.systems)
    return dy
```

**This works today with:**
- Plain Python `do_step()`
- `do_step()` that calls `@njit` decorated helpers
- Any mix of the above

**No changes needed!**

---

### ❌ `dynode/containers.py` - No changes
**Why:**
- Containers stay in Python (they don't go through JIT)
- Users extract values before calling `@njit` functions

**Pattern:**
```python
# Container in Python (unchanged)
self.states.x = 0.0

# Extract to primitive for Numba
x_val = self.states.x  # Python float

# Call JIT function with primitives
result = self._compute_jit(x_val)  # @njit function

# Assign back to container
self.ders.dx = result  # Python assignment
```

**No container modifications required!**

---

### ❌ `dynode/recorder.py` - No changes
**Why:** Observers run in Python (outside the ODE solver's hot loop)

---

### ❌ `dynode/__init__.py` - No changes (maybe)
**Options:**

**Option A: No change at all**
- Users import directly: `from numba import njit`
- Or: `from dynode.numba_utils import numba_rhs`

**Option B: Optional export (tiny change)**
```python
# Only if we want convenience
try:
    from .numba_utils import numba_rhs
    __all__.append('numba_rhs')
except ImportError:
    pass  # Numba not installed, skip
```

**Recommendation:** Option A (no change). Users who want Numba can import it.

---

## Why So Few Changes?

The key insight: **Numba JIT compilation happens at the function level, not the framework level.**

### How it works:

1. **User writes numerical function:**
   ```python
   @njit
   def _compute(x, y, mu):
       return y, mu * (1 - x**2) * y - x
   ```

2. **User calls from do_step():**
   ```python
   def do_step(self, time):
       dx, dy = self._compute(self.states.x, self.states.y, self.inputs.mu)
       self.ders.dx = dx
       self.ders.dy = dy
   ```

3. **dynode framework is unchanged:**
   - Calls `do_step(time)` as normal
   - Doesn't know or care that `_compute` is JIT compiled
   - Everything works transparently

### What gets JIT compiled:
- ✅ User's numerical computation (inside `_compute`)

### What stays in Python:
- ❌ Container access (`self.states.x`)
- ❌ Connection execution (topological sort, callbacks)
- ❌ Observers (print, record, etc.)
- ❌ System hierarchy traversal
- ❌ State dispatch/collection

**This separation is perfect!** The numerical computation (hot loop) gets compiled, everything else (cold path) stays flexible.

---

## Required Changes Summary

| File | Change Type | Effort |
|------|-------------|--------|
| `dynode/numba_utils.py` | **NEW** (optional helpers) | 2 hours |
| `dynode/__init__.py` | None (or tiny export) | 5 min |
| `dynode/system.py` | **None** | - |
| `dynode/simulation.py` | **None** | - |
| `dynode/containers.py` | **None** | - |
| `dynode/recorder.py` | **None** | - |

**Total core code changes: 2 hours (one new optional file)**

---

## What Actually Takes Time?

If the core changes are minimal, where does the 2-week estimate come from?

### Week 1: Implementation & Examples
- ✅ Write `numba_utils.py` (2 hours)
- ✅ Convert 3-5 example systems to Numba pattern (1 day)
- ✅ Benchmark examples (plain vs Numba) (1 day)
- ✅ Write test cases (ensure numerical equivalence) (1 day)
- ✅ Update `requirements.txt` / `setup.py` for optional dependency (1 hour)

### Week 2: Documentation
- ✅ User guide: "Accelerating Systems with Numba" (2 days)
- ✅ Migration guide for existing systems (1 day)
- ✅ Performance expectations documentation (1 day)
- ✅ Troubleshooting guide (common nopython errors) (1 day)

**Most effort is examples, tests, and docs - not core changes!**

---

## Dependency Management

### Option A: Optional Dependency (Recommended)
```python
# setup.py
extras_require = {
    'numba': ['numba>=0.56.0'],
    'dev': [...],
}
```

**Installation:**
```bash
pip install dynode          # Without Numba
pip install dynode[numba]   # With Numba
```

**Import handling:**
```python
# In user code
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback to plain Python
```

### Option B: Core Dependency
```python
# setup.py
install_requires = ['numpy', 'scipy', 'numba']
```

**Pros:** Users get performance by default
**Cons:** Larger install, may not work on all platforms

**Recommendation:** Option A (optional) for now.

---

## Testing Strategy

### Unit Tests (No New Infrastructure)
```python
# test/test_numba_systems.py (NEW)
from numba import njit
from dynode import SystemInterface

class VanDerPolNumba(SystemInterface):
    @staticmethod
    @njit
    def _compute(x, y, mu):
        return y, mu * (1 - x**2) * y - x

    def do_step(self, time):
        self.ders.dx, self.ders.dy = self._compute(...)

def test_numba_numerical_equivalence():
    """Ensure Numba version matches plain Python"""
    plain = VanDerPolPlain()
    numba = VanDerPolNumba()

    # Step both
    plain.do_step(0.0)
    numba.do_step(0.0)

    # Compare results
    assert abs(plain.ders.dx - numba.ders.dx) < 1e-10
    assert abs(plain.ders.dy - numba.ders.dy) < 1e-10
```

**Uses existing test infrastructure!**

---

## Migration Path for Existing Systems

### Step 1: Identify compute-intensive systems
- Look for systems with >5 states
- Systems with complex math (trig, exp, sqrt)
- Systems called in long simulations

### Step 2: Extract numerical kernel
```python
# Before
def do_step(self, time):
    self.ders.dx = complex_expression(self.states.x, self.inputs.k)

# After
@staticmethod
@njit
def _compute(x, k):
    return complex_expression(x, k)

def do_step(self, time):
    self.ders.dx = self._compute(self.states.x, self.inputs.k)
```

### Step 3: Test equivalence
```python
# Run simulation with both versions, compare results
```

### Step 4: Benchmark
```python
# Measure speedup, ensure it's worthwhile
```

**Time per system: 5-10 minutes**

---

## Conclusion

**Numba integration for dynode requires almost no core changes.**

The pattern is:
1. Users opt-in by using `@njit` on their numerical functions
2. dynode's existing architecture already supports this
3. We add optional helpers to make the pattern easier
4. We provide documentation and examples

**The beauty of this approach:**
- ✅ Zero breaking changes
- ✅ Backward compatible
- ✅ Users can migrate gradually
- ✅ Works with all existing dynode features
- ✅ Minimal maintenance burden

**Total implementation time: 2 weeks**
- Core code: 2 hours (one new file)
- Examples/tests: 4 days
- Documentation: 4 days

**Next step:** Implement `numba_utils.py` and test with existing examples.
