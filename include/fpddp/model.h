#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · model.h
//
//  The OCP model as type-erased AnyAny interfaces (no std::function / virtual):
//    AnyDynamics — continuous dynamics f(x,u,t) with Jacobians f_x, f_u.
//    AnyCost     — discrete per-stage cost + gradient + (Gauss-Newton) Hessian,
//                  and a terminal cost + gradient + Hessian.
//  Matrices use the integrator's small-dense irk::Matrix; vectors use std::span.
// ─────────────────────────────────────────────────────────────────────────────
#include <span>
#include <vector>

#include "fpddp/anyany.h"
#include "irk/essence/algebra.hpp"

namespace fpddp {

using Mat   = irk::essence::Matrix<double>;
using CSpan = std::span<const double>;
using Span  = std::span<double>;

namespace dyn {
anyany_method(rhs, (const & self, double t, CSpan x, CSpan u, Span dx) requires(self.rhs(t, x, u, dx)) -> void);
anyany_method(fx,  (const & self, double t, CSpan x, CSpan u, Mat & A) requires(self.fx(t, x, u, A))  -> void);
anyany_method(fu,  (const & self, double t, CSpan x, CSpan u, Mat & B) requires(self.fu(t, x, u, B))  -> void);
}  // namespace dyn

using AnyDynamics = aa::any_with<aa::move, aa::copy, dyn::rhs, dyn::fx, dyn::fu>;

namespace cost {
anyany_method(stage,      (const & self, double t, CSpan x, CSpan u) requires(self.stage(t, x, u)) -> double);
anyany_method(stage_grad, (const & self, double t, CSpan x, CSpan u, Span lx, Span lu) requires(self.stage_grad(t, x, u, lx, lu)) -> void);
anyany_method(stage_hess, (const & self, double t, CSpan x, CSpan u, Mat & Q, Mat & R, Mat & S) requires(self.stage_hess(t, x, u, Q, R, S)) -> void);
anyany_method(term,       (const & self, CSpan x) requires(self.term(x)) -> double);
anyany_method(term_grad,  (const & self, CSpan x, Span lx) requires(self.term_grad(x, lx)) -> void);
anyany_method(term_hess,  (const & self, CSpan x, Mat & Q) requires(self.term_hess(x, Q)) -> void);
}  // namespace cost

using AnyCost = aa::any_with<aa::move, aa::copy, cost::stage, cost::stage_grad, cost::stage_hess,
                             cost::term, cost::term_grad, cost::term_hess>;

// column-major flatten of a small dense matrix, for hpipm's OCP-QP setters.
inline std::vector<double> colmajor(const Mat& X) {
    std::vector<double> v;
    v.reserve(X.rows() * X.cols());
    for (std::size_t j = 0; j < X.cols(); ++j)
        for (std::size_t i = 0; i < X.rows(); ++i) v.push_back(X(i, j));
    return v;
}

}  // namespace fpddp
