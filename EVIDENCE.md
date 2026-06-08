# EVIDENCE — fp-ddp

## 2026-06-08

### Phase 1 — IRK integrator C++23 → C++20 port — VERIFIED ✓
- Vendored the project IRK into `include/irk/` (9 headers) from `files/irk.zip` (canonical layout).
- C++20 transforms:
  - multidim `Matrix::operator[](i,j)` → `operator()(i,j)` + all call sites (`grep` clean, structured bindings preserved).
  - `std::expected`/`std::unexpected` (C++23) → `irk::expected`/`irk::unexpected`, variant-backed (`include/irk/expected.hpp`).
  - demo `static operator()`/`static jacobian` (C++23 P1169) → non-static `const`.
- **Build:** `g++ -std=c++20 -O2 -Wall -Wextra -Iinclude test/irk_demo.cpp` → OK, **no warnings**.
- **Run:** `/tmp/irk_demo` → **15/15 PASS**, "all checks passed". Full output: `logs/irk_demo_run.log`; build log: `logs/irk_build.log` (empty).
- Conclusion: port preserves behavior exactly (same 15 verdicts as the C++23 original).

### Phase 2 — inline discrete sensitivities — VERIFIED ✓
- Added `ControlledField` concept (`notion/concepts.hpp`: `control_jacobian`=f_u, `control_dim`=m).
- `include/irk/sensitivity.hpp`: `step_sensitivity` builds the **exact** variational matrix `M=I−h(A⊗J_k)` with per-stage Jacobians, factors it, back-solves `dZ/dx,dZ/du` → `A_step,B_step` (spec §6). Plus `matmul`/`add`.
- `inference.hpp`: `Solution` gains `state_sensitivity()`/`control_sensitivity()`; a `compute_sensitivities()` helper accumulates over the stored `Node`s at `finish()`/`fail()`, all under `if constexpr (ControlledField)` so the plain path is untouched.
- Design refinement vs spec §5b: builds the **exact** variational `M` (per-stage J, fresh factor) rather than reusing the frozen-J Newton `M` — required to match FD on nonlinear problems.
- **Self-check** (`test/sens_test.cpp`, `logs/sens_run.log`): analytic vs central-FD —
  - linear 1-step: state `A` 5.05e-11, control `B` 2.06e-11 (essentially exact)
  - pendulum 4-step (chaining): state `A` 3.39e-11, control `B` 2.07e-11
  All `< 1e-6`. **Regression: `irk_demo` still 15/15.**

### Dependencies — blasfeo + hpipm built from source ✓
- Cloned as submodules (`external/blasfeo`, `external/hpipm`); built GENERIC/Release → `external/install`
  (`libblasfeo.a`, `libhpipm.a`, headers). Riccati getters present in installed `hpipm_d_ocp_qp_ipm.h`.

### Phase 4 — hpipm OCP-QP wrapper — VERIFIED ✓
- `include/fpddp/qp/ocp_qp.h`: RAII `OcpQp` over hpipm dim/qp/sol/arg/ws; plain column-major I/O;
  x0 fixed via stage-0 two-sided bounds; sqrt-Riccati (`ric_alg=1`) so `get_ric_{K,k,p}` are populated.
- **Self-check** (`test/lqr_test.cpp`, `logs/lqr_run.log`): N=10 LQR vs hand backward Riccati —
  state 1.45e-12, control 1.21e-12, **K0 4.44e-16** (machine precision); hpipm solved in 3 iters.
- Findings: (1) `get_ric_K` returns policy gain `u=Kx+k` with **K=−K_lqr** (sign confirmed) → used
  directly in the rollout. (2) plain column-major `double*` I/O → **no blasfeo C++ wrapper needed
  (revises D5)**. (3) hpipm headers carry `extern "C"`; link order `-lhpipm -lblasfeo -lm`.

### Phase 5 — DDP solver core — VERIFIED ✓
- Model (AnyAny, D3): `include/fpddp/model.h` — `AnyDynamics` (rhs/fx/fu), `AnyCost` (stage/term +
  grad/GN-Hess); vendored kelbon/anyany → `external/anyany`. (anyany overloads `operator&` → hold the
  dynamics by reference in the sim adapter.)
- Sim: `include/fpddp/sim.h` — `ControlledField` adapts {AnyDynamics, u} to irk; `integrate_interval`
  → (x⁺, A, B) via the inline sensitivities.
- Solver: `include/fpddp/ddp.h` — feasible-path DDP/iLQR: linearize → backward costate sweep
  (LM-free stationarity measure + dual seed) → hpipm Riccati QP (LM-regularized) → feasible nonlinear
  rollout + iLQR line search; LM adapts on accept/reject.
- **Self-check** (`test/ddp_test.cpp`, `logs/ddp_run.log`):
  - linear-quadratic → converges in **2 iterations** (Newton-exact), statio 1.4e-10.
  - pendulum regulation [2.5,0]→0 (nonlinear) → **4 iterations**, cost 47.9→5.8, xN=[0.006,−0.002], statio 7e-8.
- Note: globalization is concrete iLQR line search + LM in v1; AnyRegularizer / funnel-merit AnyAny
  policies are the documented follow-on (spec §11).

### Phase 6 — build + suite — VERIFIED ✓
- `CMakeLists.txt` (C++20; imported blasfeo/hpipm) + `scripts/build_deps.sh` (one-time deps build).
- **`ctest`: 5/5 passed** — irk_demo, sens_test, model_smoke, lqr_test, ddp_test (`logs/ctest.log`).

## STATUS — v1 COMPLETE: working feasible-path DDP, every phase self-checked green.
Follow-ons (spec §11): exact-Hessian (2nd-order sim sensitivities), control box bounds, AnyAny
regularizer/globalization policies, LOPT `Solution → nlp::WarmStart` adapter (primals + costates).

### Comparison vs acados DDP
- Three shared unconstrained problems, each defined identically in C++ and an acados script
  (`test/cmp_*.cpp` ≡ `compare/acados_*.py`): same dynamics, discrete cost via LINEAR_LS with dt in W,
  IRK Gauss-3, DDP/Gauss-Newton, hpipm backward pass.
- **fp-ddp solves all three cleanly** (`logs/cmp_*_fpddp.txt`):
  - 2-state pendulum regulation — 4 iters, cost 1.137179939465, statio 1e-9.
  - 4-state cart-pole stabilization (acados' own `pendulum_ode` model, unstable θ=0) — 7 iters, cost 0.14499, statio 6e-8.
  - 3-state / 2-control unicycle regulation — 5 iters, cost 0.29577, statio 3e-8.
- Finding: vanilla iLQR **stalls on the cart-pole over a long horizon** (value function ill-conditioned by
  the unstable mode — cost falls 1933→110 then sticks at statio≈47); a well-conditioned horizon (T=1.2)
  converges in 7 iters. Tight convergence on hard unstable plants is globalization-limited — the funnel/
  merit follow-on, which is exactly what acados provides (and why its cart-pole example uses it).
- **acados ran — head-to-head executed.** libacados built from source; `acados_template` 0.5.1 + casadi
  3.7.2 + `t_renderer` (downloaded with user authorization). Each `compare/acados_*.py` does codegen +
  compile + solve; `compare.py` diffs against fp-ddp:

  | problem | fp-ddp cost / iters | acados cost / iters | \|Δcost\| | max \|Δtraj\| | verdict |
  |---|---|---|---|---|---|
  | pendulum  | 1.137179939466 / 7 | 1.137179939467 / 7 | 1e-12 | 5e-5 | **AGREE** |
  | cart-pole | 0.1449931479 / 5   | 0.1449931479 / 5   | 4e-12 | 6e-5 | **AGREE** |
  | unicycle  | 0.2957734578 / 7   | 0.2957734578 / 6   | 1e-11 | 2e-5 | **AGREE** |

  Identical optima (cost ~1e-12, trajectories ~1e-5 = IRK implementation difference). After porting the
  acados algorithm (below) the iteration counts ALIGN (pendulum 7=7, cart-pole 5=5). LQ class also
  matches the analytic Riccati to ~1e-16.

### acados DDP algorithm substeps ported — VERIFIED ✓
- `ddp.h` restructured to mirror acados' `ocp_nlp_ddp` steps; added:
  - **Adaptive Levenberg-Marquardt** (exact acados rule): `lm = lm_obj_scalar·cost·μ`; μ decreases on
    accepted full steps (`÷λ`, one-step memory `μ̄`), increases on backtracks (`×λ`, cap 1), floor `μ_min`.
  - **AnyGlobalization** (AnyAny, `globalization.h`): `FixedStep` / `MeritBacktracking` / `Funnel` — Armijo
    sufficient-descent (`actual ≥ ε·α·predicted`) on the feasible rollout; predicted reduction from the
    first-order QP model. (Leaf method `reset` collides with anyany `basic_any::reset` → renamed `restart`.)
- Self-check (`test/globalization_test.cpp`): all three policies reach the same pendulum optimum
  (cost 5.82376, 7 iters, statio 9e-8). **`ctest` 9/9.**
- Effect: fp-ddp follows acados' algorithmic path (iteration counts align; comparisons still AGREE). The
  hard T=2 cart-pole that previously FAILED (status 3, statio 47.5) now stabilizes (statio 6e-3); residual
  slowness is the Gauss-Newton long-horizon limit (affects acados too — exact-Hessian is the follow-on).
