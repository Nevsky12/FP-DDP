# fp-ddp — Design Spec

- **Date:** 2026-06-08
- **Status:** Approved (brainstorming); pending implementation plan
- **Project:** `~/projects/fp-ddp` — a standalone C++20 feasible-path DDP solver
- **Ground truth:** acados `ocp_nlp_ddp.c` (+ hpipm Riccati API); vendored IRK integrator in `files/`

## 1. Purpose & scope

A faithful, general-purpose **feasible-path Differential Dynamic Programming (FP-DDP)** solver, reimplemented cleanly in C++20 on top of **blasfeo** (linear algebra) and **hpipm** (stage-QP / Riccati), using the project's own **implicit Runge–Kutta integrator** for the dynamics. The primary near-term use is as a robust **initial-guess + costate generator** for the LOPT (`~/projects/psopt`) direct-transcription IPM solver, but the solver is built as a self-contained reusable library, not a LOPT appendage.

**Why DDP.** Every DDP iterate is produced by a *nonlinear feedback rollout* of the true dynamics, so it is **dynamically feasible by construction** — the expensive property for warm-starting an interior-point method. The backward pass additionally yields the cost-to-go gradient `pᵢ`, i.e. a **costate estimate**, so a DDP pre-solve hands the consumer a feasible `(x, u)` *and* a dual seed `(π)`.

**v1 scope (this spec):** Gauss–Newton DDP (iLQR), single-phase OCP with stage-varying `nx[i], nu[i]`, **initial-state constraints only** (matching acados DDP, which hard-errors on any other stage constraint), funnel + merit globalization, Levenberg–Marquardt + Hessian regularization.

**Out of v1 (interfaces left open):** exact-Hessian DDP (second-order sensitivity propagation), control/path/terminal inequality constraints, the LOPT `Solution → nlp::WarmStart` adapter, multi-phase horizons.

## 2. Locked decisions (with rationale)

| # | Decision | Rationale |
|---|---|---|
| D1 | Faithful **general standalone** DDP | User directive; reusable beyond LOPT. |
| D2 | **C++20** throughout (integrator down-ported from C++23) | User directive; single language level across the project. |
| D3 | Model interface via **AnyAny** (`aa::any_with` + `anyany_method`), no `std::function`/virtual | Matches LOPT `nlp/`,`mocp/` idiom (their D1). |
| D4 | **Initial-state constraints only** in v1 | Faithful to acados DDP; keeps the feasible-rollout story clean. |
| D5 | **blasfeo throughout** for QP data + thin C++20 RAII wrapper; zero-copy into hpipm | User directive; performance + faithful; one LA world at the QP layer. |
| D6 | Deps via **git submodules + CMake** (`add_subdirectory`) | Self-contained, reproducible from a fresh clone. |
| D7 | **Vendor + extend the project IRK integrator**: C++20 port + **inline sensitivity** emission | User directive ("use the integrator I provided", "inline", "C++20"). |
| D8 | Stage QP solved by **hpipm IPM**; extract Riccati feedback via `d_ocp_qp_ipm_get_ric_{K,k,P,p}` | Confirmed in `acados/ocp_qp/ocp_qp_hpipm.c:425-461`; gives `K,k` for the rollout and `p` as the costate seed. |
| D9 | **Gauss–Newton (iLQR)** Hessian in v1; exact-Hessian a scoped follow-on | Second-order sensitivity through an IRK is heavy; GN needs only first-order `Aᵢ,Bᵢ`. |

## 3. Architecture

```
fp-ddp/
  CMakeLists.txt                    # C++20; add_subdirectory(external/{blasfeo,hpipm})
  external/{blasfeo,hpipm}/         # git submodules
  external/expected/                # vendored C++20 expected (tl::expected or minimal)
  include/fpddp/
    util/expected.h                 # fpddp::expected alias (C++20 std::expected substitute)
    types.h                         # f64, idx, Status enum
    blasfeo/{dmat.h,dvec.h}         # RAII C++20 wrapper over blasfeo_dmat/dvec; zero-copy views
    irk/                            # vendored integrator, C++20-ported + sensitivity-extended
      being.h essence/{algebra,orthogonal,collocation}.h
      notion/{concepts,stages}.h judgment.h inference.h irk.h
    model/
      any_dynamics.h                # AnyDynamics (anyany): rhs f(x,u,t), fx, fu
      any_cost.h                    # AnyCost (anyany): stage l + grad/Hess, terminal; GN residual adapter
      gauss_newton.h                # residual -> GN cost (Hessian = JᵀJ)
    sim/
      controlled_field.h            # {AnyDynamics, u} -> irk ControlledField (f, .jacobian=fx, .control_jacobian=fu)
      integrate_interval.h          # irk::integrate over [tᵢ,tᵢ₊₁] -> (x⁺, Aᵢ, Bᵢ)
    ocp/
      dims.h                        # N, nx[i], nu[i], nbx0
      stage_qp.h                    # per-stage blasfeo blocks: A,B,b, Q,R,S,q,r
      solution.h                    # x[],u[],pi[] (costate), cost, iter stats
    qp/
      hpipm_ocp_qp.h                # set dims/data, ipm solve, get_ric_{K,k,P,p}
    solve/
      linearize.h                   # (Aᵢ,Bᵢ,cost derivs) -> blasfeo stage QP + LM term
      any_regularizer.h             # AnyRegularizer: mirror | project | none
      rollout.h                     # feasible sweep: uᵢ=ūᵢ+αkᵢ+Kᵢ(xᵢ-x̄ᵢ), x⁺=integrate(xᵢ,uᵢ)
      any_globalization.h           # AnyGlobalization: funnel | merit | fixed_step
      ddp.h                         # Options, solve loop, termination, getters
  src/…                             # matching .cpp where not header-only
  test/                             # gtest
  examples/                         # pendulum p2p, LQR
```

**Dependency DAG (acyclic):** `blasfeo wrapper` and `irk` are foundations → `model` (leaf) → `sim` (uses `irk` + `model`) → `linearize` (uses `sim` + `model` → `stage_qp`) → `qp/hpipm` (solves `stage_qp`) → `rollout` (uses `sim` + `K,k`) → `ddp` orchestrates. Every leaf is unit-testable in isolation.

**Two LA worlds, one boundary:** the `irk` layer keeps its own small-dense `essence::Matrix`/`PLU` for the stage system (n- and sn-sized); the QP layer is blasfeo+hpipm. `linearize` is the boundary: it packs the small dense `Aᵢ (n×n)`, `Bᵢ (n×m)`, and cost blocks into blasfeo stage matrices.

## 4. Core interfaces

**AnyDynamics / AnyCost (AnyAny, mirroring `mocp/any_guess.h` style):**
```cpp
namespace dyn {
  anyany_method(rhs, (const& self, f64 t, CSpan x, CSpan u, Span dx) requires(self.rhs(t,x,u,dx)) -> void);
  anyany_method(fx,  (const& self, f64 t, CSpan x, CSpan u, Mat& A)  requires(self.fx (t,x,u,A))  -> void); // ∂f/∂x
  anyany_method(fu,  (const& self, f64 t, CSpan x, CSpan u, Mat& B)  requires(self.fu (t,x,u,B))  -> void); // ∂f/∂u
}
using AnyDynamics = aa::any_with<aa::move, aa::copy, dyn::rhs, dyn::fx, dyn::fu>;
// AnyCost: stage cost value + (l_x,l_u) + (l_xx,l_uu,l_ux), terminal cost; GN adapter fills Hess = JᵀJ.
```
A model that provides `rhs`+`fx`+`fu` is sufficient for v1 (GN). `f_xx` (exact-Hessian) is a later optional method.

**blasfeo C++20 wrapper:** RAII `DMat`/`DVec` owning a `blasfeo_dmat/dvec` (aligned alloc in ctor, free in dtor; move-only), with named ops (`gemv`, `gemm`, `pack/unpack`, submatrix views) so call sites read like math, and a raw accessor so hpipm consumes them with zero copy.

**hpipm wrapper (`qp/hpipm_ocp_qp.h`):** owns `d_ocp_qp_dim`, `d_ocp_qp`, `d_ocp_qp_sol`, `d_ocp_qp_ipm_arg`, `d_ocp_qp_ipm_ws`; sets stage data from `StageQp`; `solve()`; getters `K(i)`, `k(i)`, `pi(i)`, `ric_P(i)`, `ric_p(i)` wrapping `d_ocp_qp_ipm_get_ric_*`.

**Policies (AnyAny):** `AnyRegularizer` (`mirror` = eigenvalue mirroring, `project` = PSD projection, `none`), `AnyGlobalization` (`funnel`, `merit`, `fixed_step`), each with the small method set the DDP loop calls.

**Solution:** `x[i] (nx[i])`, `u[i] (nu[i])`, `pi[i]` (costate, from `ric_p`/QP duals), final cost, per-iteration stat table, `Status`.

## 5. Integrator extensions (D7) — precise

The integrator stays a coherent "ladder of notions"; two changes thread through it.

**(a) C++20 down-port.** The only C++23 features:
- `essence::Matrix::operator[](i,j)` (multidim subscript, `algebra.h:38`) → `operator()(i,j)`; update every call site (`algebra, orthogonal, collocation, stages, inference, judgment`).
- `std::expected` (`PLU::factor → expected<PLU,Singular>` `algebra.h:73`; `integrate → expected<Solution,Failure>` `inference.h`) → `fpddp::expected` (`util/expected.h`, vendored `tl::expected` or a minimal internal type).

Everything else (concepts, `std::span`, `std::ranges`, `constexpr`, `shared_ptr`) is already C++20.

**(b) Inline sensitivities.** Add to `notion/concepts.h` a `ControlledField` refinement that also exposes `control_jacobian` (`∂f/∂u`) and the control dimension `m`. In `notion/stages.h`, after simplified-Newton converges in `StageSolver::solve`, **reuse the existing `PLU` factorization of `M = I − h(A⊗J)`** to solve, for that step,
```
M · (dZ/dx) = h·(A ⊗ I) · blkdiag(Jⱼ)     →  s blocks  dZ/dx  (each n×n)
M · (dZ/du) = h·(A ⊗ I) · blkdiag(Bᶜⱼ)    →  s blocks  dZ/du  (each n×m)
A_step = I + Σᵢ b̃ᵢ (dZᵢ/dx)               (n×n)
B_step =     Σᵢ b̃ᵢ (dZᵢ/du)               (n×m)
```
with `Jⱼ = f_x(Yⱼ)`, `Bᶜⱼ = f_u(Yⱼ)`, `Yⱼ = y + Zⱼ`. In `inference.h`, accumulate across the internal steps of the interval: running `(Ā,B̄)` starts `(I,0)` and updates `Ā ← A_step·Ā`, `B̄ ← A_step·B̄ + B_step`; at interval end `(Aᵢ,Bᵢ) = (Ā,B̄)` is stored on `Solution`. See §6 for the derivation. Reusing the frozen-`J` `M` is the IND-consistent, factorization-reusing choice (D7); RHS Jacobians `Jⱼ,Bᶜⱼ` may be refreshed per stage for accuracy without re-factoring `M` — a calibration knob validated against the FD oracle (§9).

## 6. Sensitivity derivation (the crux)

One accepted IRK step from `y` with control `u` constant over the interval; tableau `(A, c, b̃)`, step `h`. Stage system and output:
```
Gᵢ(Z; y,u) = Zᵢ − h Σⱼ aᵢⱼ f(t+cⱼh, y+Zⱼ, u) = 0 ,   y⁺ = y + Σᵢ b̃ᵢ Zᵢ
```
Differentiate `G = 0` totally at the solution:
```
(∂G/∂Z)·(dZ/d(y,u)) + ∂G/∂(y,u) = 0
∂Gᵢ/∂Z_k = δᵢ_k I − h aᵢ_k J_k = M          (J_k = f_x(Y_k))
∂Gᵢ/∂y   = −h Σⱼ aᵢⱼ Jⱼ
∂Gᵢ/∂u   = −h Σⱼ aᵢⱼ Bᶜⱼ                       (Bᶜⱼ = f_u(Yⱼ))
⟹  M·(dZ/dy) = h·[Σⱼ a₁ⱼ Jⱼ; …; Σⱼ a_sⱼ Jⱼ]      (sn×n)
   M·(dZ/du) = h·[Σⱼ a₁ⱼ Bᶜⱼ; …]                 (sn×m)
A_step = I + Σᵢ b̃ᵢ (dZᵢ/dy),  B_step = Σᵢ b̃ᵢ (dZᵢ/du)
```
Multi-step interval (`y₀→…→y_K = x⁺`, `u` constant): `Aᵢ = A_K⋯A₁`, `Bᵢ = Σ_{k} (A_K⋯A_{k+1}) B_k`. Implemented as the running accumulation in §5(b). `M` is exactly the matrix the StageSolver already factors — hence "reuse the factorization."

## 7. Algorithm — one DDP iteration (mirrors `ocp_nlp_ddp.c`)

1. **Linearize:** for each stage, `sim` integrates `[tᵢ,tᵢ₊₁]` → `(x⁺, Aᵢ, Bᵢ)`; `cost` → gradients/GN-Hessian. Pack into the blasfeo stage QP. Add the Levenberg–Marquardt term.
2. **Regularize** the stage Hessians via `AnyRegularizer`.
3. **Residuals + termination:** KKT = stationarity (`‖∇_u H‖`) ∧ dynamic feasibility (`‖defect‖`); plus min-step, max-iter, NaN. (No inequality/complementarity terms in v1 — initial-state-only.)
4. **Backward pass:** hpipm solves the stage QP; extract `Kᵢ, kᵢ` (feedback/feedforward) and `πᵢ` (costate).
5. **Globalization / forward pass:** if the initial guess is dynamically infeasible, take one **full-step feasible rollout** to feasibilize; otherwise run `funnel`/`merit` line search over `α`, each trial being a **feasible nonlinear rollout** `uᵢ = ūᵢ + α kᵢ + Kᵢ(xᵢ − x̄ᵢ)`, `xᵢ₊₁ = integrate(xᵢ, uᵢ)`.
6. Loop.

Robustness rests on exactly three pillars: the **feasible rollout**, **LM + Hessian regularization**, and the **line search**.

## 8. Error handling & status

- Integrator boundary returns `fpddp::expected`; an `irk::Failure` (step underflow / attempt budget / Newton failure) bubbles up as `Status::integration_breakdown` with the failing stage/time.
- Solver `Status`: `success`, `max_iter`, `min_step`, `qp_failure` (mapped from hpipm status), `nan_detected`, `integration_breakdown`.
- No exceptions across the public API for control flow; `expected`/`Status` only. Allocation failures and contract violations may assert/throw.

## 9. Testing strategy

- **Leaf units:** blasfeo wrapper ops vs reference; `sim` inline sensitivities **finite-difference-checked** (the FD step-map difference is the oracle for `Aᵢ,Bᵢ`); hpipm `get_ric_{K,k}` vs a hand-coded Riccati on a small LQR; rollout produces a dynamically feasible trajectory for any `α`.
- **Integration:** LQR (DDP converges in one step; matches the analytic Riccati solution); pendulum swing-up / point-to-point vs an acados-DDP reference trajectory; feasibility invariant (every accepted iterate dynamically feasible); convergence-order sanity.
- **Port regression:** the integrator's own demo suite (`files/demo.cpp`, 15 PASS verdicts) ported to C++20 must still pass — guards the C++23→C++20 down-port.

## 10. Build & dependencies

- C++20, GCC 13+ (the integrator uses `std::format` in its demo; library is `std::format`-free).
- `external/blasfeo`, `external/hpipm` as git submodules, built via `add_subdirectory`; `external/expected` vendored (header-only).
- Top-level CMake builds the `fpddp` library (mostly header-only), `test/` (gtest), `examples/`.
- One-command build from a fresh clone (`git submodule update --init` + `cmake --build`).

## 11. Phasing

- **v1 (this spec):** GN-DDP, initial-state constraints, funnel+merit, LM+regularize, inline first-order sensitivities, hpipm backward pass. Tests: LQR + pendulum + sensitivity FD checks + ported integrator demo.
- **Follow-on A:** exact-Hessian DDP — second-order sensitivity propagation through the integrator (adjoint-contracted), `f_xx` on the model.
- **Follow-on B:** control box bounds — hpipm constrained stage QP + clipped/projected rollout.
- **Follow-on C:** LOPT adapter — `fpddp::Solution → lopt::nlp::WarmStart` (primals + `π` costates), respecting the cross-mesh transfer rules already documented for LOPT.

## 12. Open questions / risks

- **Sensitivity accuracy vs reuse:** frozen-`J` `M` (cheapest, reuses factorization) vs per-stage `Jⱼ` in the RHS (more accurate). Default to reuse + per-stage RHS Jacobians; validate against FD; expose as a knob.
- **hpipm unconstrained path:** confirm hpipm's IPM on an essentially-unconstrained OCP reduces to ~1 Riccati factorization and that `get_ric_*` is populated in that regime (acados relies on it; verify on the LQR test early).
- **Adaptive-step sensitivities:** sensitivities are computed holding the accepted node structure fixed (standard); confirm acceptable for guess generation, or pin `fixed_step` per shooting interval when exactness matters.
- **`u` parametrization over the interval:** v1 holds `u` constant per shooting interval (piecewise-constant control); higher-order control parametrization is out of scope.
```
