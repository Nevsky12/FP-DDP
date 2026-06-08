#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · globalization.h
//
//  Globalization policies as AnyAny, ported from acados' DDP globalizations.
//  The solver runs the backtracking loop + (feasible) rollout and asks the
//  policy whether a trial is acceptable:
//    accept(c0, ct, pred, alpha, h0, ht) -> bool
//      c0,ct : current / trial objective       pred : model-predicted reduction
//      alpha : step size                        h0,ht: current / trial dynamic infeasibility
//  restart(h0) at the start of a solve; notify(alpha, ht) after an accepted step
//  (the funnel updates its width there).  For a feasible-rollout DDP ht≈0, so
//  merit and funnel both reduce to the f-type Armijo on the objective.
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>

#include "fpddp/anyany.h"

namespace fpddp {

namespace glob {
anyany_method(accept, (const & self, double c0, double ct, double pred, double alpha, double h0, double ht) requires(self.accept(c0, ct, pred, alpha, h0, ht)) -> bool);
anyany_method(restart,  (& self, double h0) requires(self.restart(h0)) -> void);
anyany_method(notify, (& self, double alpha, double ht) requires(self.notify(alpha, ht)) -> void);
}  // namespace glob

using AnyGlobalization = aa::any_with<aa::move, aa::copy, glob::accept, glob::restart, glob::notify>;

// Full step, no globalization (accept the first finite trial — i.e. alpha = 1).
struct FixedStep {
    bool accept(double, double ct, double, double, double, double) const { return std::isfinite(ct); }
    void restart(double) {}
    void notify(double, double) {}
};

// Armijo sufficient-descent backtracking (acados merit_backtracking; merit ≈ cost
// for a feasible-rollout DDP): accept iff actual_reduction ≥ ε·α·predicted.
struct MeritBacktracking {
    double eps = 1e-4;
    bool accept(double c0, double ct, double pred, double alpha, double, double) const {
        if (!std::isfinite(ct)) return false;
        return (c0 - ct) >= eps * alpha * std::max(0.0, pred - 1e-9);
    }
    void restart(double) {}
    void notify(double, double) {}
};

// Funnel globalization (acados ocp_nlp_globalization_funnel): a shrinking funnel
// on the dynamic infeasibility with f-type (objective Armijo) and h-type
// (feasibility) steps.  For a feasible rollout (ht≈0) this is the f-type Armijo,
// with the h-type path retained for completeness.
struct Funnel {
    double eps = 1e-4, suff_dec = 0.9, frac_switch = 1e-3, init_ub = 1.0, init_inc = 10.0;
    double width = 1e10;
    void restart(double h0) { width = std::max(init_ub, init_inc * h0); }
    bool accept(double c0, double ct, double pred, double alpha, double, double ht) const {
        if (!std::isfinite(ct)) return false;
        if (ht > width) return false;                                   // outside funnel
        if (alpha * std::max(0.0, pred) >= frac_switch * ht || ht < 1e-12) {
            if ((c0 - ct) >= eps * alpha * std::max(0.0, pred - 1e-9)) return true;   // f-type Armijo
        }
        if (ht <= suff_dec * width) return true;                        // h-type feasibility step
        return false;
    }
    void notify(double, double ht) { if (ht <= suff_dec * width) width = std::max(suff_dec * width, ht); }
};

}  // namespace fpddp
