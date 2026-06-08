# PLAN — fp-ddp

Faithful general **C++20 feasible-path DDP** on blasfeo + hpipm, using the project **IRK integrator**.
Full design + rationale: `docs/superpowers/specs/2026-06-08-fp-ddp-design.md`.

## Current phase
**v1 COMPLETE — all phases done & verified (ctest 5/5 green).** fp-ddp solves OCPs by feasible-path
DDP on hpipm/blasfeo + the project IRK with inline sensitivities. Per-phase self-checks in EVIDENCE.md.
Follow-ons: exact-Hessian, control bounds, AnyAny regularizer/globalization, LOPT WarmStart adapter.

## Build order (each phase ends with a self-check)
1. IRK C++20 port — demo 15/15 ........................... ✓ DONE (EVIDENCE 2026-06-08)
2. Inline sensitivities (ControlledField + dZ/dx,dZ/du) + FD check
3. blasfeo C++20 RAII wrapper + unit test
4. hpipm OCP-QP wrapper + LQR Riccati check
5. model / sim / linearize / rollout / globalization / ddp loop
6. integration tests (LQR one-step, pendulum p2p) + CMake + submodules

## Invariant
Every phase must build `-std=c++20 -Wall -Wextra` clean and pass its self-check before the next.
