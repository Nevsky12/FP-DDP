# fp-ddp

A standalone **C++20 feasible-path Differential Dynamic Programming** solver on
**hpipm** (Riccati backward pass) + **blasfeo**, using a vendored variable-order
**implicit Runge–Kutta integrator** (Gauss–Legendre / Radau IIA) for the dynamics
with **inline discrete sensitivities**. Built to generate robust, dynamically
feasible initial guesses — primals *and* costates — for interior-point trajectory
optimization (e.g. LOPT).

## Status

**v1 complete.** Solves optimal-control problems with initial-state constraints
(Gauss–Newton Hessian). Every component is self-checked (`ctest`: 5/5):
the integrator port (15/15 demo), inline sensitivities (vs finite differences),
the hpipm wrapper (vs hand Riccati), and end-to-end DDP (Newton-exact on an LQR,
nonlinear pendulum regulation). See `EVIDENCE.md` for per-phase evidence and
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
- `include/fpddp/` — `model.h` (AnyAny dynamics/cost), `sim.h`, `qp/ocp_qp.h` (hpipm), `ddp.h`
- `test/` — `irk_demo`, `sens_test`, `lqr_test`, `ddp_test`, `model_smoke`
- `external/` — `blasfeo`, `hpipm` (git submodules), `anyany` (vendored header lib)

## Follow-ons

Exact-Hessian DDP (2nd-order sim sensitivities), control box bounds, AnyAny
regularizer/globalization policies, and a LOPT `Solution → nlp::WarmStart` adapter.
