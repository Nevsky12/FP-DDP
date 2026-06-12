#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · globalization.h
//
//  Globalization policies as AnyAny, ported from acados (v0.5.4):
//    FixedStep         — acados ocp_nlp_globalization_fixed_step: take
//                        alpha = step_length unconditionally (no acceptance
//                        test; a NaN iterate is caught by the solver's NaN
//                        termination, as in acados).
//    MeritBacktracking — acados ocp_nlp_ddp_backtracking_line_search (the
//                        DDP-specific Armijo in
//                        ocp_nlp_globalization_merit_backtracking.c): accept iff
//                          ct − c0 ≤ min(−ε·α·max(pred_qp, 0) + 1e-18, 0),
//                        pred_qp = −(optimal QP objective) — i.e. the model
//                        reduction including the ½ dᵀHd term (LM-regularized).
//    Funnel            — acados ocp_nlp_globalization_funnel f-/h-type logic.
//                        acados only supports the funnel with SQP; it is kept
//                        here for feasible-path use where ht ≡ 0 makes it an
//                        f-type Armijo on pred_lin = −gᵀd (the first-order
//                        model acados uses as predicted_optimality_reduction).
//                        The h-type test is an *else* branch of the switching
//                        condition (as in acados); the penalty ('b'/'p') mode is
//                        omitted — it is unreachable when ht ≡ 0.
//
//  Solver protocol: restart(h0) once per solve with the initial L1 dynamic
//  infeasibility; per trial accept(c0, ct, pred_qp, pred_lin, alpha, h0, ht)
//  where h0/ht are the current/trial infeasibility (identically 0 for the
//  feasible rollout); alpha0() is the line-search starting step.
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>

#include "fpddp/anyany.h"

namespace fpddp {

namespace glob {
anyany_method(accept, (& self, double c0, double ct, double pred_qp, double pred_lin, double alpha, double h0, double ht) requires(self.accept(c0, ct, pred_qp, pred_lin, alpha, h0, ht)) -> bool);
anyany_method(restart, (& self, double h0) requires(self.restart(h0)) -> void);
anyany_method(alpha0,  (const & self) requires(self.alpha0()) -> double);
}  // namespace glob

using AnyGlobalization = aa::any_with<aa::move, aa::copy, glob::accept, glob::restart, glob::alpha0>;

// acados fixed_step: alpha = step_length, no acceptance test.
struct FixedStep {
    double step_length = 1.0;
    double alpha0() const { return step_length; }
    bool accept(double, double, double, double, double, double, double) { return true; }
    void restart(double) {}
};

// acados DDP merit backtracking (Armijo on the objective; trial iterates are
// feasible so the merit function is the cost).  eps default = the acados
// Python-interface default for DDP (globalization_eps_sufficient_descent).
struct MeritBacktracking {
    double eps = 1e-6;
    double alpha0() const { return 1.0; }
    bool accept(double c0, double ct, double pred_qp, double, double alpha, double, double) {
        const double negative_ared = ct - c0;                       // NaN ct ⇒ false ⇒ backtrack
        return negative_ared <= std::min(-eps * alpha * std::max(pred_qp, 0.0) + 1e-18, 0.0);
    }
    void restart(double) {}
};

// acados funnel (f-/h-type steps).  Switching compares the predicted objective
// reduction against the predicted infeasibility reduction (= h0 for a step that
// removes the linearized defect); the h-type branch is only reachable when the
// switching condition FAILS, and shrinks the funnel by the kappa rule.
struct Funnel {
    double eps = 1e-6, suff_dec = 0.9, kappa = 0.9, frac_switch = 1e-3;
    double init_ub = 1.0, init_inc = 15.0;
    double width = 1.0;
    double alpha0() const { return 1.0; }
    void restart(double h0) { width = std::max(init_ub, init_inc * h0); }
    bool accept(double c0, double ct, double, double pred_lin, double alpha, double h0, double ht) {
        if (std::isnan(ct)) return false;
        if (ht > width) return false;                               // outside funnel
        if (alpha * pred_lin >= frac_switch * h0) {                 // switching condition
            return (c0 - ct) >= eps * alpha * std::max(0.0, pred_lin - 1e-9);  // f-type Armijo
        } else if (ht <= suff_dec * width) {                        // h-type (else-branch, as in acados)
            width = (1.0 - kappa) * ht + kappa * width;             // acados decrease_funnel
            return true;
        }
        return false;  // (penalty mode omitted: unreachable for ht ≡ 0)
    }
};

}  // namespace fpddp
