#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · fd_jacobian.h
//
//  Central finite-difference Jacobians of a dynamics rhs, so a model with messy
//  analytic derivatives (cart-pole, …) can supply only rhs and get fx/fu for
//  free.  ~1e-9 accurate — fine for DDP linearization.  (For production accuracy
//  prefer analytic or AD Jacobians; this keeps the comparison models compact.)
// ─────────────────────────────────────────────────────────────────────────────
#include <span>
#include <vector>

#include "fpddp/model.h"

namespace fpddp {

template <class Rhs>
void fd_state_jac(const Rhs& rhs, double t, CSpan x, CSpan u, Mat& A, double eps = 1e-6) {
    const std::size_t n = x.size();
    std::vector<double> xp(x.begin(), x.end()), fp(n), fm(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double s = xp[j];
        xp[j] = s + eps; rhs(t, CSpan(xp), u, Span(fp));
        xp[j] = s - eps; rhs(t, CSpan(xp), u, Span(fm));
        xp[j] = s;
        for (std::size_t i = 0; i < n; ++i) A(i, j) = (fp[i] - fm[i]) / (2 * eps);
    }
}

template <class Rhs>
void fd_control_jac(const Rhs& rhs, double t, CSpan x, CSpan u, Mat& B, double eps = 1e-6) {
    const std::size_t n = x.size(), m = u.size();
    std::vector<double> up(u.begin(), u.end()), fp(n), fm(n);
    for (std::size_t j = 0; j < m; ++j) {
        const double s = up[j];
        up[j] = s + eps; rhs(t, x, CSpan(up), Span(fp));
        up[j] = s - eps; rhs(t, x, CSpan(up), Span(fm));
        up[j] = s;
        for (std::size_t i = 0; i < n; ++i) B(i, j) = (fp[i] - fm[i]) / (2 * eps);
    }
}

}  // namespace fpddp
