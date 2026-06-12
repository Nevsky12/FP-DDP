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

## 2026-06-12

### acados-parity fix pass (line-by-line audit vs acados v0.5.4-10) — VERIFIED ✓
Audit of `ocp_nlp_ddp.c` + `ocp_nlp_common.c` + globalization modules found one real bug and a set
of mismatches vs the acados internals; all fixed:
- **Funnel logic bug (fixed):** the h-type test was reachable when the switching condition held, and
  with ht≡0 (feasible rollouts) it accepted *every* trial — the funnel degenerated to FixedStep, even
  accepting cost-increasing steps. Now the h-type branch is an `else` of the switching condition
  (acados structure), the switching condition compares against the predicted infeasibility reduction,
  the width shrinks by the kappa rule `(1−κ)·ht + κ·w` inside the h-accept, and defaults align
  (init_inc 15, ε 1e-6). Note: acados forbids DDP+funnel (SQP-only); the policy is kept for
  feasible-path use where it reduces to the f-type Armijo.
- **Predicted reduction:** now `pred_qp = −(optimal QP objective)` incl. the ½dᵀHd term with the
  LM-regularized Hessian (acados `ocp_nlp_compute_qp_objective_value`); the funnel uses the
  first-order `pred_lin = −gᵀd` (acados `predicted_optimality_reduction`). Was first-order only.
- **Armijo formula:** exact acados DDP rule `ct−c0 ≤ min(−ε·α·max(pred,0)+1e-18, 0)`.
- **Residuals & termination:** acados-style `res_stat` (Lagrange gradient over all (u,x) blocks with
  the carried QP duals — full-step dual update) + `res_eq`; termination in `check_termination` order:
  NaN → (eq ∧ stat) → zero-residual LS (`tol_zero_res`, default 0 = acados' effective value — the C
  field is never initialized upstream) → small step (`step_norm < tol_eq`) → maxiter, with the
  residual check also run *at* max_iter (acados `eval_residual_at_max_iter`, DDP default).
- **Infeasible initial guess:** new `solve(x0, x_guess, u_guess, opt)` accepts an arbitrary primal
  guess and takes acados' forced full feedback step at iteration 0 (the acados default init x≡x0,
  u≡0 is exactly this case).
- **Defaults aligned to the acados Python interface for DDP+merit:** ε_suff_descent 1e-6,
  α_min 1e-17, α_reduction 0.7, max_iter 100, tol_stat/tol_eq 1e-6, LM off by default
  (levenberg_marquardt 0.0), hpipm BALANCE mode + iter_max 50; per-stage `lm_scaling` option added
  (acados multiplies the LM term by the cost `scaling`). QP MAXITER is tolerated (only hard QP
  failures abort), matching acados. `FixedStep` gained `step_length` and accepts unconditionally.
- Failed integration steps now NaN-fill the state (sim.h) so they surface via the NaN termination
  instead of silently returning the unchanged state. `ctest` 9/9 after the pass.

### Stress test vs acados — 10 problems, per-iteration parity — VERIFIED ✓
`test/stress_fpddp.cpp` ≡ `compare/stress_acados.py` (same dynamics/weights/N/dt/x0, LINEAR_LS with
dt in W and scaling=1, IRK GL3 ×1 with acados newton_tol 1e-12, DDP + merit + adaptive LM,
tol 1e-6, acados-default infeasible init x≡x0, u≡0 on both sides); `compare/stress_compare.py`
diffs status, iteration count, and the per-iteration cost/LM/res_stat sequences. Driver:
`scripts/stress.sh`. Result — **9/10 strict AGREE, iterate-for-iterate**:

| problem | iters fp = ac | iter-cost rel | LM seq | traj |
|---|---|---|---|---|
| pendulum | 7 = 7 | 3.0e-12 | exact | exact (10 dec) |
| dintegrator (LQ) | 4 = 4 | 1.9e-13 | exact | exact |
| vdp | 7 = 7 | 1.5e-13 | exact | exact |
| unicycle | 6 = 6 | 2.2e-12 | exact | exact |
| cartpole | 5 = 5 | 5.0e-12 | 2e-15 | 2.2e-09 |
| cartpole_long (T=2) | 5 = 5 | 5.1e-11 | 2e-14 | 3.6e-09 |
| quadrotor (6-state) | 7 = 7 | 2.2e-12 | exact | exact |
| pendulum_hard | 9 = 9 | 3.0e-13 | exact | exact |
| **vdp_hard (backtracks α=0.7)** | 17 = 17 | 1.7e-11 | 1e-12 | 1.0e-10 |
| cartpole_swing (knife-edge) | 59 vs 11 | 2.5e-10 (shared 12 its) | 1e-11 | — informational |

- The previously-stalling long-horizon cart-pole now converges in 5 iterations on both sides — the
  acados-default x≡x0 init (forced-step path) conditions the unstable plant far better than the old
  open-loop u=0 rollout init.
- cartpole_swing: parity holds for 12 iterations *including three matched backtracking decisions*
  (res_stat 4.6400208303 vs 4.6400208302 at the split point); beyond that, marginal accept/reject
  flips amplify integrator-level differences (acados' 20-capped IRK Newton vs the vendored adaptive
  IRK at violent trial states) and the paths legitimately separate — acados exits MINSTEP, fp-ddp
  converges (status 0, 59 its). Algorithm parity is established by the shared prefix.

### IRK bench vs acados sim_irk — performance & robustness — MEASURED ✓
`test/bench_irk.cpp` ≡ `compare/bench_irk_acados.py`: 186 single GL3 (s=3) steps with forward
sensitivities over mild→violent state grids (pendulum, cartpole, VdP μ=5), Newton tol 1e-12 both
sides (ours ≤10 simplified-Newton iters, frozen J/step, scaled-norm + contraction forecast,
*explicit failure*; acados ≤20 full-Newton iters, fresh J/iter, unscaled ‖ΔK‖∞, *no failure
status*). Oracle: CVODES at 1e-13. Table: `logs/bench_irk_table.txt`.
- **Speed** (median/step incl. sensitivities, both-converged cases): acados 3.0–4.0 µs vs ours
  4.0–7.4 µs → acados 1.0–1.8× faster (blasfeo + codegen'd model, no per-call allocation; our
  cartpole number also carries central-FD Jacobians).
- **Accuracy**: identical max error vs CVODES wherever both converge (same discrete map; columns
  equal to all printed digits) — differences are pure GL3 discretization error (1e-9 @ dt=0.05 →
  1.5e-3 @ dt=0.5).
- **Robustness**: ours flagged 75/186 (loud, NaN-fill → the DDP line search rejects cleanly); ~45
  of those acados' stronger full Newton solved fine (our simplified Newton is over-conservative),
  but on the ~30 cases beyond both, acados returned **silent garbage with status SUCCESS** — 38
  silent failures total, including scaled errors of 0.1–0.21 (cartpole ω=40, dt=0.2) and **7
  NaN-as-SUCCESS** returns (VdP dt=0.5); on vdp[13] ours converged properly where acados NaN'd.
  This is the mechanism behind the cartpole_swing stress split.

### IRK/Newton hardening + speed — beats acados sim_irk on both axes — VERIFIED ✓
`include/fpddp/irk_stepper.h`: persistent stepper replacing the per-call irk driver.  The Butcher
tableau is cached (it was 52% — 2.1 µs — of every old call), all workspace persists, the first
substep warm-starts from the previous call, line-search rollouts skip the variational solves
(`want_sens=false`), and the stage solve escalates:  simplified Newton (frozen J) → cold-predictor
retry → **full Newton** (per-stage Jacobians refreshed each iteration — acados' scheme, but
convergence-checked in the scaled norm) → substep halving → loud NaN failure.  The DDP loop holds
one warm stepper per shooting interval.  Re-run of the 186-case bench (all Jacobians analytic,
FD-validated at startup; same grids; `logs/bench_irk_table.txt`):

| group | ours cold | ours warm | acados | speedup | fpFAIL | acados bad |
|---|---|---|---|---|---|---|
| pendulum .05/.2/.5 | 1.15/1.36/1.63 µs | ≈cold | 3.0/3.0/3.75 µs | 2.3–2.6× | 0 (was 0/0/12) | 0 |
| cartpole .04/.2 | 3.73/4.69 µs | ≈cold | 4.0/5.0 µs | 1.07× | 0 (was 14/35) | 0 |
| vdp .1/.5 | 1.72/1.73 µs | ≈cold | 4.0/5.0 µs | 2.3–2.9× | 0 (was 6/8) | **8 NaN-as-SUCCESS** |

- **0/186 failures, 0 substep splits needed** (the full-Newton fallback recovered every previously
  flagged case); every accepted step still verified to the 1e-12 scaled tolerance — zero silent.
- Identical collocation roots to acados on all 178 cases where acados converges; on the 8 VdP
  dt=0.5 cases acados returns NaN with status SUCCESS while ours converges properly.
- `ctest` 10/10 and the 10-problem DDP stress parity vs acados unchanged after the rewiring.

### Exact-Hessian DDP, automatically computed — VERIFIED ✓
`HessianMode::exact_fd` (ddp.h): the QP Hessian gains the dynamics curvature Σ_k π_k ∂²Φ_k/∂(x,u)²,
computed by central differences of the exact adjoint sensitivities g(z) = [A(z)ᵀπ; B(z)ᵀπ] — no
user second derivatives; multipliers are the carried QP duals (empirically better-conditioned than
bare adjoint costates).  Validated against a double-FD oracle of πᵀΦ(z): max diff 2.2e-7.  Key
finding: the exact **stage-block** Hessian is legitimately indefinite even at the optimum (only the
Riccati-reduced Hessian must be PSD) — projecting it costs the convergence rate (VdP: linear at
ratio 0.63, 25 its).  It is therefore handed to hpipm **raw**; eigenvalue projection
(`regularize.h`, mirror/project via Jacobi) runs only if the QP actually fails, and a hybrid
GN→exact switch (`exact_switch_stat`, default 1e-1) keeps the far-from-solution phase on GN.
Self-check (`test/exact_hessian_test.cpp`, ctest 10/10):
- LQ: exact ≡ GN (1 = 1 iteration, same optimum to 1e-10).
- hard pendulum (tol 1e-9): exact **3 its, statio 7.3e-13** vs GN 4 its, 1.1e-10 — quadratic tail.
- stiff VdP μ=3 (tol 1e-7): exact **10 its, statio 3.8e-12** vs GN 16 its, 7.1e-08; the tail goes
  2.0e-3 → 7.9e-4 → 2.6e-7 → 3.8e-12 (textbook quadratic).
