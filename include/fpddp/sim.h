#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · sim.h
//
//  Shooting-interval integration built on the project IRK integrator.  A
//  ControlledField adapts {AnyDynamics, fixed u} to irk's ControlledField
//  concept; integrate_interval integrates [t0,t1] and returns x⁺ together with
//  the discrete sensitivities A=∂x⁺/∂x, B=∂x⁺/∂u (computed inline by irk).
// ─────────────────────────────────────────────────────────────────────────────
#include <span>
#include <vector>

#include "irk/irk.hpp"
#include "fpddp/model.h"

namespace fpddp {

// Adapts {dynamics, control held over the interval} to irk's ControlledField.
struct ControlledField {
    const AnyDynamics& dyn;
    std::span<const double> u;
    std::size_t m;
    void operator()(double t, std::span<const double> x, std::span<double> dx) const { dyn.rhs(t, x, u, dx); }
    void jacobian(double t, std::span<const double> x, irk::Matrix<double>& J) const { dyn.fx(t, x, u, J); }
    void control_jacobian(double t, std::span<const double> x, irk::Matrix<double>& B) const { dyn.fu(t, x, u, B); }
    std::size_t control_dim() const { return m; }
};

struct StepResult {
    std::vector<double> xnext;
    Mat A, B;          // ∂x⁺/∂x (n×n), ∂x⁺/∂u (n×m)
    bool ok = true;
};

// Integrate one shooting interval with u held constant; nsub fixed Gauss s=3
// substeps keep the step map smooth and the variational sensitivities exact.
inline StepResult integrate_interval(const AnyDynamics& dyn, std::span<const double> x,
                                     std::span<const double> u, double t0, double t1,
                                     std::size_t nsub) {
    ControlledField f{dyn, u, u.size()};
    irk::Options<double> opt;
    opt.family       = irk::Family::gauss_legendre;
    opt.fixed_step   = (t1 - t0) / double(nsub);
    opt.fixed_stages = 3;

    auto sol = irk::integrate(f, irk::Interval<double>{t0, t1}, x, irk::Measure<double>{1e-12, 1e-12}, opt);

    StepResult r;
    r.ok = bool(sol);
    if (!sol) {
        r.xnext.assign(x.begin(), x.end());
        r.A = Mat::identity(x.size());
        r.B = Mat(x.size(), u.size());
        return r;
    }
    r.xnext.assign(sol->y_end().begin(), sol->y_end().end());
    r.A = sol->state_sensitivity();
    r.B = sol->control_sensitivity();
    return r;
}

}  // namespace fpddp
