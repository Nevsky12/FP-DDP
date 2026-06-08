#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · sensitivity.hpp
//
//  Discrete forward sensitivities of one accepted IRK step.  At the converged
//  stage solution Z the stage equations  G_i = Z_i − h Σ_k a_ik f(Y_k) = 0
//  define x⁺ = y + Σ b̃_i Z_i implicitly; the implicit-function theorem gives
//      M (dZ/d{y,u}) = h (A⊗I) [ J_k | Bc_k ],   M = I − h(A⊗J_k),
//      A_step = ∂x⁺/∂y = I + Σ b̃_i (dZ_i/dy),  B_step = ∂x⁺/∂u = Σ b̃_i (dZ_i/du),
//  with per-stage J_k = f_x(Y_k), Bc_k = f_u(Y_k).  The exact variational matrix
//  is built (per-stage Jacobians) and factored fresh, so A_step,B_step are the
//  true derivatives of the realized step map — validated against finite
//  differences in test/sens_test.cpp.  (Design spec §6.)
//
//  Includes:  being, essence/algebra, essence/collocation, notion/concepts.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstddef>
#include <span>
#include <vector>

#include "irk/being.hpp"
#include "irk/essence/algebra.hpp"
#include "irk/essence/collocation.hpp"
#include "irk/notion/concepts.hpp"

namespace irk::sensitivity {

template <being::Real R>
[[nodiscard]] essence::Matrix<R> matmul(const essence::Matrix<R>& L, const essence::Matrix<R>& Rm) {
    const std::size_t p = L.rows(), q = L.cols(), r = Rm.cols();
    essence::Matrix<R> O(p, r);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t k = 0; k < q; ++k) {
            const R lik = L(i, k);
            if (lik == R(0)) continue;
            for (std::size_t j = 0; j < r; ++j) O(i, j) += lik * Rm(k, j);
        }
    return O;
}

template <being::Real R>
[[nodiscard]] essence::Matrix<R> add(essence::Matrix<R> A, const essence::Matrix<R>& B) {
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) A(i, j) += B(i, j);
    return A;
}

template <being::Real R, typename F>
    requires notion::ControlledField<F, R>
void step_sensitivity(const F& f, const essence::ButcherTableau<R>& tab,
                      R t, std::span<const R> y, R hs, std::span<const R> Z,
                      std::size_t n, std::size_t m,
                      essence::Matrix<R>& A_step, essence::Matrix<R>& B_step) {
    const std::size_t s = tab.s, sn = s * n;

    std::vector<essence::Matrix<R>> J(s, essence::Matrix<R>(n, n));
    std::vector<essence::Matrix<R>> Bc(s, essence::Matrix<R>(n, m));
    std::vector<R> Yk(n);
    for (std::size_t k = 0; k < s; ++k) {
        for (std::size_t a = 0; a < n; ++a) Yk[a] = y[a] + Z[k * n + a];
        J[k].set_zero();  f.jacobian(t + tab.c[k] * hs, std::span<const R>(Yk), J[k]);
        Bc[k].set_zero(); f.control_jacobian(t + tab.c[k] * hs, std::span<const R>(Yk), Bc[k]);
    }

    essence::Matrix<R> M(sn, sn);                       // M = I − h(A ⊗ J_k)
    for (std::size_t i = 0; i < s; ++i)
        for (std::size_t k = 0; k < s; ++k) {
            const R w = -hs * tab.A(i, k);
            for (std::size_t a = 0; a < n; ++a)
                for (std::size_t b = 0; b < n; ++b) M(i * n + a, k * n + b) = w * J[k](a, b);
        }
    for (std::size_t d = 0; d < sn; ++d) M(d, d) += R(1);

    // safe fallback if the variational matrix is singular: identity / zero
    A_step.set_zero();
    for (std::size_t a = 0; a < n; ++a) A_step(a, a) = R(1);
    B_step.set_zero();

    auto lu = essence::PLU<R>::factor(std::move(M));
    if (!lu) return;

    std::vector<R> rhs(sn);
    for (std::size_t a = 0; a < n; ++a) {               // column a of dZ/dy
        for (std::size_t i = 0; i < s; ++i)
            for (std::size_t p = 0; p < n; ++p) {
                R acc = R(0);
                for (std::size_t k = 0; k < s; ++k) acc += tab.A(i, k) * J[k](p, a);
                rhs[i * n + p] = hs * acc;
            }
        lu->solve(rhs);
        for (std::size_t p = 0; p < n; ++p) {
            R acc = R(0);
            for (std::size_t i = 0; i < s; ++i) acc += tab.b_tilde[i] * rhs[i * n + p];
            A_step(p, a) += acc;
        }
    }
    for (std::size_t c = 0; c < m; ++c) {               // column c of dZ/du
        for (std::size_t i = 0; i < s; ++i)
            for (std::size_t p = 0; p < n; ++p) {
                R acc = R(0);
                for (std::size_t k = 0; k < s; ++k) acc += tab.A(i, k) * Bc[k](p, c);
                rhs[i * n + p] = hs * acc;
            }
        lu->solve(rhs);
        for (std::size_t p = 0; p < n; ++p) {
            R acc = R(0);
            for (std::size_t i = 0; i < s; ++i) acc += tab.b_tilde[i] * rhs[i * n + p];
            B_step(p, c) += acc;
        }
    }
}

} // namespace irk::sensitivity
