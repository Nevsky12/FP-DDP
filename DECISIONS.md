# DECISIONS — fp-ddp

Authoritative design + rationale: `docs/superpowers/specs/2026-06-08-fp-ddp-design.md` (§2 table).

Summary: C++20; **AnyAny** model interface (no `std::function`/virtual); **blasfeo throughout** for the
QP layer + thin C++20 RAII wrapper; **hpipm** IPM stage QP / Riccati backward pass (`get_ric_{K,k,P,p}`);
**vendored IRK** integrator (C++20 port + **inline** sensitivities); **initial-state constraints only** (v1);
**Gauss–Newton** Hessian (v1, exact-Hessian later); deps via **git submodules + CMake**.

## Implementation notes (discovered during build)
- **C++20 `expected` substitute:** variant-backed `irk::expected`/`irk::unexpected`
  (`include/irk/expected.hpp`) — covers the used surface: value/unexpected ctor, contextual bool,
  `operator*`, `operator->`, `.error()`. Chosen over vendoring tl::expected to keep the integrator
  self-contained.
- **Sensitivities reuse the StageSolver `M` factorization** (IND-consistent); RHS stage Jacobians may be
  refreshed per stage for accuracy — validated against an FD oracle.
- **Shell is zsh:** `status` is a read-only variable — use `rc` (or similar) in scripts.
