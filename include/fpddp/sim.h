#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · sim.h
//
//  Shooting-interval integration. integrate_interval integrates [t0,t1] with u
//  held constant and returns x⁺ together with the discrete sensitivities
//  A=∂x⁺/∂x, B=∂x⁺/∂u, via the persistent hardened IrkStepper (cached tableau,
//  simplified→full Newton escalation, substep halving on breakdown).  Warm-start
//  state is dropped between calls here, so the map is a pure function of
//  (x, u, t0, t1, nsub); callers wanting cross-call warm starting (the DDP
//  loop) hold their own IrkStepper instances.
//
//  On genuine integration failure xnext is NaN-filled (ok=false) so the
//  failure propagates loudly into costs/residuals — never a silent value.
// ─────────────────────────────────────────────────────────────────────────────
#include <map>
#include <span>
#include <utility>
#include <vector>

#include "fpddp/irk_stepper.h"
#include "fpddp/model.h"

namespace fpddp {

struct StepResult {
    std::vector<double> xnext;
    Mat A, B;          // ∂x⁺/∂x (n×n), ∂x⁺/∂u (n×m)
    bool ok = true;
};

// Integrate one shooting interval with u held constant; nsub fixed Gauss s=3
// substeps (the stepper may split further on Newton breakdown).
inline StepResult integrate_interval(const AnyDynamics& dyn, std::span<const double> x,
                                     std::span<const double> u, double t0, double t1,
                                     std::size_t nsub) {
    thread_local std::map<std::pair<std::size_t, std::size_t>, IrkStepper> steppers;
    auto& st = steppers.try_emplace({x.size(), u.size()}, x.size(), u.size()).first->second;
    st.forget();                                       // deterministic: no cross-call state

    StepResult r;
    r.ok = st.step(dyn, x, u, t0, t1, nsub, r.xnext, r.A, r.B);
    return r;
}

}  // namespace fpddp
