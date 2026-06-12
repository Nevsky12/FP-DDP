# fp-ddp

A standalone **C++20 feasible-path Differential Dynamic Programming** solver on
**hpipm** (Riccati backward pass) + **blasfeo**, using a vendored variable-order
**implicit Runge–Kutta integrator** (Gauss–Legendre / Radau IIA) for the dynamics
with **inline discrete sensitivities**. Built to generate robust, dynamically
feasible initial guesses — primals *and* costates — for interior-point trajectory
optimization (e.g. LOPT).

## Status

**v1 complete + acados-parity verified + faster/more-robust IRK + exact Hessian.**
Solves optimal-control problems with initial-state constraints. The solver
mirrors acados' `ocp_nlp_ddp` (v0.5.4) step for step — adaptive
Levenberg-Marquardt, acados residuals/termination, QP-objective predicted
reduction, merit/funnel/fixed-step globalization, infeasible-initial-guess
forced step — validated **iterate for iterate** against acados on 10 problems
(per-iteration cost matching to ~1e-12, identical iteration counts;
`scripts/stress.sh`). The hardened IRK stepper (cached tableau, warm starts,
simplified→full Newton→substep-halving escalation, loud failure) is **1.1–2.9×
faster than acados' sim_irk per step and strictly more robust**: 0/186 bench
failures with every step verified to 1e-12, where acados silently returns
unconverged values and even NaN-as-SUCCESS (8 cases). Optional
**automatically-computed exact-Hessian DDP** (FD of adjoint sensitivities, raw
indefinite Hessians to hpipm, on-demand projection) gives textbook quadratic
tails: VdP 10 its / 3.8e-12 vs GN 16 its / 7.1e-08. Every component is
self-checked (`ctest`: 10/10). See `EVIDENCE.md` for per-phase evidence and
`docs/superpowers/specs/2026-06-08-fp-ddp-design.md` for the design.

## Build

```sh
git submodule update --init        # blasfeo, hpipm
scripts/build_deps.sh              # build them once -> external/install
cmake -S . -B build && cmake --build build -j
( cd build && ctest --output-on-failure )
```

Requires GCC 13+ (C++20).

## Layout

- `include/irk/` — vendored IRK integrator (C++20 port + inline sensitivities)
- `include/fpddp/` — `model.h` (AnyAny dynamics/cost), `irk_stepper.h` (hardened persistent
  stepper), `sim.h`, `qp/ocp_qp.h` (hpipm), `regularize.h` (on-demand PSD projection),
  `globalization.h` (FixedStep | MeritBacktracking | Funnel), `ddp.h`
- `test/` — `irk_demo`, `sens_test`, `lqr_test`, `ddp_test`, `model_smoke`,
  `globalization_test`, `exact_hessian_test`, `cmp_*` (acados head-to-head),
  `stress_fpddp` (10-problem parity driver), `bench_irk` (IRK perf/robustness bench)
- `compare/` — acados-side scripts + per-iteration diff harness (`scripts/stress.sh` runs it all)
- `external/` — `blasfeo`, `hpipm` (git submodules), `anyany` (vendored header lib)

## Follow-ons

Control box bounds, an embedded-error-driven accuracy split in the stepper, a
reduced-Hessian-aware regularizer, and a LOPT `Solution → nlp::WarmStart` adapter.
