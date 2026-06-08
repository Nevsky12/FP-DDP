# TODO — fp-ddp

## Phase 1 — IRK C++20 port  (DONE)
- [x] Vendor irk into `include/irk/`; canonical layout from `files/irk.zip`
- [x] Port multidim subscript + `std::expected`; `include/irk/expected.hpp`
- [x] Build `-std=c++20` clean; `test/irk_demo.cpp` → 15/15 PASS

## Phase 2 — inline sensitivities  (DONE)
- [x] `ControlledField` concept in `notion/concepts.hpp` (`control_jacobian`=f_u, `control_dim`=m)
- [x] `sensitivity.hpp`: exact variational `M` → `dZ/dx,dZ/du` → `A_step,B_step`
- [x] `inference.hpp`: `compute_sensitivities()` chains over Nodes → `Solution` sensitivities (if constexpr)
- [x] FD self-check: linear exact (~5e-11), pendulum chaining (~3e-11); demo still 15/15

## Phase 3 — blasfeo C++20 wrapper  (SUPERSEDED)
- [x] Not needed — hpipm OCP interface is plain column-major double*; blasfeo stays internal (D5 revised)

## Phase 4 — hpipm OCP-QP wrapper  (DONE)
- [x] Built blasfeo+hpipm from source → external/install
- [x] `OcpQp` RAII wrapper; ipm solve; `get_ric_{K,k,p}`
- [x] LQR cross-check vs hand Riccati: traj ~1e-12, K0 ~4e-16 (logs/lqr_run.log)

## Phase 5 — solver core  (DONE)
- [x] `AnyDynamics`/`AnyCost` (AnyAny); `sim` (controlled_field, integrate_interval)
- [x] linearize (+LM) + hpipm stage QP; costate adjoint sweep (stationarity + dual seed)
- [x] feasible rollout + iLQR line search; `ddp` loop + termination
- [x] **acados DDP substeps ported**: adaptive LM + AnyGlobalization (FixedStep/Merit/Funnel) + Armijo
      predicted-reduction line search (`globalization.h`, restructured `ddp.h`); globalization_test + 9/9 ctest

## Phase 6 — tests + build  (DONE)
- [x] LQR (Newton-exact, 2 iters) + pendulum regulation (4 iters, →origin); ddp_test
- [x] CMake + submodules + build_deps.sh; ctest 5/5 (logs/ctest.log)

## Follow-ons (spec §11)
- [ ] exact-Hessian DDP (2nd-order sim sensitivities)
- [ ] control box bounds (hpipm constrained stage QP + clipped rollout)
- [ ] LOPT `fpddp::Solution → lopt::nlp::WarmStart` adapter (primals + costates)
