#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · essence/collocation.hpp                                      [layer 1]
//
//  ESSENCE III (Wesen): actuality.  One generative operation —
//      integration_weights(c, θ):   Σ_j w_j c_j^{k-1} = θ^k / k,  k = 1..s
//  i.e. the exact quadrature of polynomials over [0, θ] on the nodes c —
//  unfolds, by varying θ alone, into the whole tableau:
//      rows of A (θ = c_i),  b (θ = 1),  the embedded defect d,
//      and the dense-output weights (any θ = σ).
//  The Family enum is Chelpanov's division of the genus into species.
//
//  Exports:   integration_weights, Family, ButcherTableau.
//  Includes:  being, essence/algebra, essence/orthogonal.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "irk/being.hpp"
#include "irk/essence/algebra.hpp"
#include "irk/essence/orthogonal.hpp"

namespace irk::essence {

// Weights of the interpolatory quadrature  ∫₀^θ p ≈ Σ w_j p(c_j),
// exact for polynomials of degree < s.  Solved as the dual (moment-side)
// Vandermonde system by the Björck–Pereyra algorithm: O(s²) and forward
// stable for the increasing positive nodes produced by this library.
template <being::Real R>
[[nodiscard]] std::vector<R> integration_weights(std::span<const R> c, R theta) {
    const std::size_t s = c.size();
    std::vector<R> w(s);
    R p = R(1);
    for (std::size_t k = 0; k < s; ++k) {        // moments θ^{k+1}/(k+1)
        p *= theta;
        w[k] = p / R(k + 1);
    }
    for (std::size_t k = 0; k + 1 < s; ++k)      // phase 1: synthetic division
        for (std::size_t i = s - 1; i > k; --i)
            w[i] -= c[k] * w[i - 1];
    for (std::size_t k = s - 1; k-- > 0;) {      // phase 2: divided differences
        for (std::size_t i = k + 1; i < s; ++i)
            w[i] /= (c[i] - c[i - 1 - k]);
        for (std::size_t i = k; i + 1 < s; ++i)
            w[i] -= w[i + 1];
    }
    return w;
}

// Division of the concept: two species of collocation.
enum class Family { gauss_legendre, radau_iia };

[[nodiscard]] constexpr const char* name(Family f) noexcept {
    return f == Family::gauss_legendre ? "Gauss" : "Radau IIA";
}

template <being::Real R>
struct ButcherTableau {
    Family family{Family::radau_iia};
    std::size_t s{0};            // stage count
    unsigned order{0};           // classical order: Gauss 2s, Radau IIA 2s-1
    unsigned controller_order{0};// order of the embedded error estimate: s
    std::vector<R> c;            // nodes
    std::vector<R> b;            // quadrature weights, θ = 1
    std::vector<R> d;            // embedded defect  d = b - b̂
    std::vector<R> b_tilde;      // A^{-T} b : update    y₁ = y₀ + Σ b̃ᵢ Zᵢ
    std::vector<R> d_tilde;      // A^{-T} d : estimate  est = Σ d̃ᵢ Zᵢ
    Matrix<R> A;                 // collocation matrix
    PLU<R> luAT;                 // factorization of Aᵀ
    bool stiffly_accurate{false};
    bool filter_estimate{false};

    [[nodiscard]] static ButcherTableau make(Family family, std::size_t s) {
        if (s < 1 || s > 16)
            throw std::logic_error("irk: stage count must lie in [1, 16]");

        ButcherTableau t;
        t.family = family;
        t.s      = s;
        t.c      = (family == Family::gauss_legendre) ? gauss_nodes<R>(s)
                                                      : radau_iia_nodes<R>(s);
        t.order  = (family == Family::gauss_legendre) ? unsigned(2 * s)
                                                      : unsigned(2 * s - 1);
        t.controller_order = unsigned(s);

        const std::span<const R> cs(t.c);
        t.A = Matrix<R>(s, s);
        for (std::size_t i = 0; i < s; ++i) {
            const auto row = integration_weights<R>(cs, t.c[i]);
            for (std::size_t j = 0; j < s; ++j) t.A[i, j] = row[j];
        }
        t.b = integration_weights<R>(cs, R(1));

        t.d = t.b;                                    // d = b - b̂, b̂ on c₁..c_{s-1}
        if (s > 1) {
            const auto bh = integration_weights<R>(cs.first(s - 1), R(1));
            for (std::size_t j = 0; j + 1 < s; ++j) t.d[j] -= bh[j];
        }

        Matrix<R> At(s, s);
        for (std::size_t i = 0; i < s; ++i)
            for (std::size_t j = 0; j < s; ++j) At[i, j] = t.A[j, i];
        auto lu = PLU<R>::factor(std::move(At));
        if (!lu) throw std::logic_error("irk: collocation matrix is singular");
        t.luAT = std::move(*lu);

        t.b_tilde = t.b;  t.luAT.solve(t.b_tilde);
        t.d_tilde = t.d;  t.luAT.solve(t.d_tilde);

        t.stiffly_accurate = (family == Family::radau_iia);
        if (t.stiffly_accurate) {                     // snap b̃ = e_s exactly
            for (auto& v : t.b_tilde) v = R(0);
            t.b_tilde.back() = R(1);
        }
        t.filter_estimate = (family == Family::radau_iia);
        return t;
    }

    // Dense-output weights: u(σ) = y₀ + Σ wᵢ(σ) Zᵢ with w(σ) = A^{-T}·∫₀^σ.
    // σ = 1 reproduces b̃, hence the trajectory is continuous by construction.
    [[nodiscard]] std::vector<R> output_weights(R sigma) const {
        auto w = integration_weights<R>(std::span<const R>(c), sigma);
        luAT.solve(w);
        return w;
    }

    // Worst violation of the simplifying moment conditions C(s) and B(s):
    // a direct certificate of construction accuracy in the working precision.
    [[nodiscard]] R order_condition_residual() const {
        R worst = 0;
        std::vector<R> pw(s, R(1));                   // c_j^{k-1}
        for (unsigned k = 1; k <= s; ++k) {
            R sb = 0;
            for (std::size_t j = 0; j < s; ++j) sb += b[j] * pw[j];
            worst = std::max(worst, std::abs(sb - R(1) / R(k)));
            for (std::size_t i = 0; i < s; ++i) {
                R sa = 0;
                for (std::size_t j = 0; j < s; ++j) sa += A[i, j] * pw[j];
                R target = R(1) / R(k);
                for (unsigned m = 0; m < k; ++m) target *= c[i];
                worst = std::max(worst, std::abs(sa - target));
            }
            for (std::size_t j = 0; j < s; ++j) pw[j] *= c[j];
        }
        return worst;
    }
};

} // namespace irk::essence
