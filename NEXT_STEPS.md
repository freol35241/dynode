# Next Steps: Dynode Acceleration

Based on the comprehensive investigation of JAX and Numba integration, here are the recommended next steps.

---

## Immediate Actions (This Week)

### 1. Decision: Numba vs JAX vs Both

**Recommendation: Start with Numba, add JAX later if needed**

**Rationale:**
- ✅ Numba = Quick win (2 weeks vs 3 weeks)
- ✅ Minimal user disruption (opt-in decorators)
- ✅ Validates acceleration strategy with low risk
- ✅ JAX can be added later as separate module

**Decision criteria:**
- If >80% of users need only CPU acceleration → **Numba only**
- If GPU/autodiff is critical for >30% users → **Both Numba + JAX**
- If uncertain → **Start with Numba, gather feedback**

---

### 2. Run Prototypes with Real dynode

**Tasks:**
```bash
# Install dependencies
pip install numba  # For Numba prototype
pip install jax jaxlib diffrax  # For JAX prototype

# Run benchmarks
cd examples/
python numba_prototype.py
python jax_prototype.py

# Analyze results:
# - What are actual speedups on representative systems?
# - Is compilation overhead acceptable?
# - Do benefits justify implementation effort?
```

**Success criteria:**
- Numba: >3x average speedup on medium/large systems
- JAX: >10x speedup on GPU (if available)
- Compilation overhead: <5s for typical systems

---

### 3. User Survey (Optional but Recommended)

Create a brief survey for dynode users:

1. What size are your typical systems? (# of states)
2. How long do your simulations run? (seconds/minutes/hours)
3. Do you have GPU access?
4. Do you need automatic differentiation (parameter optimization)?
5. Would you use Numba acceleration (minimal code changes)?
6. Would you use JAX (functional API, but GPU support)?

**Use results to prioritize:**
- High CPU usage, no GPU → **Numba priority**
- GPU available, optimization needs → **JAX priority**
- Mixed → **Both**

---

## Short-Term Implementation (Next 1-2 Months)

### Option A: Numba Integration (2 weeks)

**Week 1: Core Implementation**
- [ ] Create `dynode/numba_utils.py` helper module
- [ ] Design decorator pattern: `@numba_system` or similar
- [ ] Convert 3 test systems to Numba pattern
- [ ] Benchmark against plain Python versions
- [ ] Validate numerical accuracy (match scipy results)

**Week 2: Documentation & Polish**
- [ ] Write user guide: "Accelerating Systems with Numba"
- [ ] Add 5 examples (simple → complex)
- [ ] Create troubleshooting guide (common nopython errors)
- [ ] Update installation docs (`pip install dynode[numba]`)
- [ ] Add CI tests for Numba compatibility

**Deliverable:** `dynode` v0.5.0 with optional Numba support

---

### Option B: JAX Module (3 weeks)

**Week 1: Core Infrastructure**
- [ ] Create `dynode/jax/` submodule
- [ ] Implement `JaxSystemInterface` base class
- [ ] Implement `JaxSimulation` with Diffrax backend
- [ ] Test single VanDerPol system (CPU)
- [ ] Test on GPU (if available)

**Week 2: Composition & Features**
- [ ] Design subsystem composition pattern
- [ ] Implement functional connection mechanism
- [ ] Add transformation utilities (jit, vmap, grad wrappers)
- [ ] Test hierarchical systems
- [ ] Benchmark vs scipy and Numba

**Week 3: Documentation & Examples**
- [ ] Write migration guide: "From classic dynode to dynode.jax"
- [ ] Create examples:
  - Basic: VanDerPol on GPU
  - Advanced: Parameter optimization with grad
  - Power user: Batch simulations with vmap
- [ ] API reference documentation
- [ ] Performance guide (when to use GPU)

**Deliverable:** `dynode` v0.6.0 with `dynode.jax` module

---

### Option C: Both (Recommended)

**Phase 1: Numba (2 weeks)** - See Option A
**Phase 2: Gather feedback (2 weeks)** - Monitor usage, collect requests
**Phase 3: JAX if needed (3 weeks)** - See Option B

**Total time:** 5-7 weeks with feedback loop

---

## Long-Term Strategy (6-12 Months)

### Scenario 1: Numba Success, JAX Not Needed

**If:** Numba provides sufficient speedup for >90% of users

**Actions:**
- Make Numba default recommendation in docs
- Consider making Numba a core dependency (not optional)
- Focus optimization efforts on Numba patterns
- Monitor for GPU requests → add JAX only if demand grows

---

### Scenario 2: JAX Critical for Power Users

**If:** >20% of users need GPU or autodiff

**Actions:**
- Promote JAX module prominently in docs
- Provide rich examples for ML/optimization workflows
- Consider JAX-first architecture for future versions
- Maintain Numba for backward compatibility

---

### Scenario 3: Dual Support is Optimal

**If:** Users split between CPU-only and GPU/optimization needs

**Actions:**
- Document decision matrix: "Which backend should I use?"
- Maintain both Numba and JAX with equal priority
- Ensure feature parity where possible
- Provide conversion tools (classic → Numba → JAX)

---

## Technical Debt & Maintenance

### If Implementing Numba:

**Ongoing costs:**
- Keep up with Numba releases (2-3 updates/year)
- Test against new Python versions
- Help users debug nopython mode errors
- **Estimated effort:** 1-2 days/quarter

### If Implementing JAX:

**Ongoing costs:**
- Keep up with JAX/Diffrax releases (monthly updates)
- Maintain functional API separately from classic API
- Help users with functional programming patterns
- Monitor GPU compatibility (CUDA versions)
- **Estimated effort:** 2-3 days/quarter

### Both:

- Ensure numerical equivalence across backends
- Coordinate feature additions (new solver, callbacks, etc.)
- **Estimated effort:** 3-5 days/quarter

---

## Risk Mitigation

### Risk 1: Limited Speedup in Practice

**Probability:** Medium (30%)
**Impact:** High (wasted effort)

**Mitigation:**
- ✅ Run prototypes on real user systems before committing
- ✅ Set success criteria (>3x average speedup)
- ✅ Be willing to abandon if benchmarks don't justify effort

---

### Risk 2: User Confusion (Two APIs)

**Probability:** Medium (if doing both)
**Impact:** Medium (support burden)

**Mitigation:**
- ✅ Clear decision matrix in docs
- ✅ Separate documentation sections
- ✅ Examples for both patterns
- ✅ Migration tools

---

### Risk 3: Maintenance Burden

**Probability:** Low (well-tested libraries)
**Impact:** Medium (time sink)

**Mitigation:**
- ✅ Comprehensive test suite
- ✅ Pin dependency versions
- ✅ Automated CI for compatibility
- ✅ Community contributions (accept PRs for optimizations)

---

## Success Metrics

### For Numba:
- [ ] >3x average speedup on test suite
- [ ] >50% of medium/large systems use Numba within 6 months
- [ ] <5 GitHub issues related to Numba compatibility/year
- [ ] Positive user feedback

### For JAX:
- [ ] >10x speedup on GPU for large systems
- [ ] Autodiff enables new use cases (parameter fitting, etc.)
- [ ] >10% of users adopt JAX module within 6 months
- [ ] Research papers cite dynode.jax for ML applications

---

## Final Recommendation

### Recommended Path: **Start with Numba**

1. **Week 1-2:** Implement Numba prototype (full implementation)
2. **Week 3:** Run benchmarks, gather user feedback
3. **Week 4:** Decision point:
   - If successful (>3x speedup) → Release v0.5.0
   - If unsuccessful → Document findings, defer acceleration
4. **Month 2-3:** Monitor usage, collect GPU/autodiff requests
5. **Month 3-4:** Implement JAX if demand justifies effort

**Estimated total effort:**
- Numba only: **2 weeks**
- Numba + JAX: **5 weeks** (staggered)
- Maintenance: **1-3 days/quarter**

**Expected ROI:**
- User time saved: 100s-1000s of hours/year (faster simulations)
- Positioning: dynode as "fast-by-default" ODE framework
- Ecosystem fit: Enables ML/scientific computing workflows

---

## How to Get Started

### This Week:

1. **Review prototypes:** Read through `examples/numba_prototype.py` and `examples/jax_prototype.py`

2. **Run benchmarks:** If dynode is fully set up, execute prototypes and measure real speedups

3. **Make decision:** Based on benchmarks and user needs, choose Numba, JAX, or both

4. **Create GitHub issue:** "Acceleration strategy: [Numba|JAX|Both]" with decision rationale

5. **Start implementation:** Follow roadmap above

### Questions to Answer:

- Do current users complain about performance? (Measure demand)
- Are there GPU users in the community? (JAX priority)
- Is parameter optimization a common use case? (Autodiff need)
- What's the typical system size? (Small → Numba, Large → JAX)

---

## Resources

- **Investigation reports:** See `ACCELERATION_INVESTIGATION.md`
- **Prototypes:** See `examples/numba_prototype.py` and `examples/jax_prototype.py`
- **Numba docs:** https://numba.readthedocs.io/
- **JAX docs:** https://jax.readthedocs.io/
- **Diffrax docs:** https://docs.kidger.site/diffrax/

---

**Author:** Claude (acceleration investigation)
**Date:** 2025-11-28
**Branch:** `claude/dynode-jax-investigation-01VRW5pTVdHZ4Q9VmP7VEdz7`
**Status:** Investigation complete, awaiting decision
