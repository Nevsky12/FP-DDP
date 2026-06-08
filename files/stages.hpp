#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · notion/stages.hpp                                            [layer 2]
//
//  NOTION II–III (Begriff): judgment and syllogism.  The stage equations
//      Z_i = h Σ_j a_ij f(t + c_j h, y + Z_j)
//  predicate consistency of the stages; the simplified Newton iteration is
//  the inference that restores it — major premise the frozen Jacobian (the
//  law), minor premise the residual (the fact), conclusion the correction.
//
//  Exports:   NewtonControls, NewtonOutcome, NewtonReport, StageSolver.
//  Includes:  being, essence/algebra, essence/collocation, notion/concepts.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include "irk/being.hpp"
#include "irk/essence/algebra.hpp"
#include "irk/essence/collocation.hpp"
#include "irk/notion/concepts.hpp"

namespace irk::notion {

struct NewtonControls {
    std::size_t max_iterations{10};
    double tolerance{1e-2};      // κ: required ratio of Newton error to the unit error
};

enum class NewtonOutcome {
    converged, singular_matrix, diverged, slow, nonfinite, max_iterations
};

[[nodiscard]] constexpr const char* name(NewtonOutcome o) noexcept {
    switch (o) {
        case NewtonOutcome::converged:       return "converged";
        case NewtonOutcome::singular_matrix: return "singular matrix";
        case NewtonOutcome::diverged:        return "diverged";
        case NewtonOutcome::slow:            return "slow contraction";
        case NewtonOutcome::nonfinite:       return "non-finite increment";
        case NewtonOutcome::max_iterations:  return "iteration limit";
    }
    return "?";
}

template <being::Real R>
struct NewtonReport {
    NewtonOutcome outcome{NewtonOutcome::max_iterations};
    std::size_t iterations{0};
    std::size_t nfev{0};
    std::size_t factorizations{0};
    R contraction{0};            // last θ = ‖Δ_k‖ / ‖Δ_{k-1}‖
    R increment{0};              // last scaled ‖Δ‖

    [[nodiscard]] bool converged() const noexcept { return outcome == NewtonOutcome::converged; }
};

// Resolves the stage displacements Z (stage-major: Z = [Z₁; …; Z_s]) of one
// step attempt.  The Jacobian is refreshed once per step point and reused
// across step-size retries; the iteration matrix M = I - h(A ⊗ J) is
// refactored on every attempt.
template <being::Real R>
class StageSolver {
public:
    explicit StageSolver(std::size_t n)
        : n_(n), J_(n, n), ywork_(n), f0work_(n), fwork_(n) {}

    template <typename F>
        requires VectorField<F, R>
    [[nodiscard]] std::size_t refresh_jacobian(const F& f, R t, std::span<const R> y) {
        return jacobian<R>(f, t, y, J_, ywork_, f0work_, fwork_);
    }

    [[nodiscard]] const essence::Matrix<R>& jac() const noexcept { return J_; }

    template <typename F>
        requires VectorField<F, R>
    [[nodiscard]] NewtonReport<R> solve(const F& f, const essence::ButcherTableau<R>& tab,
                                        R t, std::span<const R> y, R hs, std::span<R> Z,
                                        const being::Measure<R>& meas,
                                        const NewtonControls& ctl) {
        const std::size_t s = tab.s, m = s * n_;
        NewtonReport<R> rep;

        essence::Matrix<R> M(m, m);                       // M = I - h (A ⊗ J)
        for (std::size_t i = 0; i < s; ++i)
            for (std::size_t j = 0; j < s; ++j) {
                const R w = -hs * tab.A[i, j];
                if (w == R(0)) continue;
                for (std::size_t a = 0; a < n_; ++a)
                    for (std::size_t b = 0; b < n_; ++b)
                        M[i * n_ + a, j * n_ + b] = w * J_[a, b];
            }
        for (std::size_t k = 0; k < m; ++k) M[k, k] += R(1);

        rep.factorizations = 1;
        auto lu = essence::PLU<R>::factor(std::move(M));
        if (!lu) { rep.outcome = NewtonOutcome::singular_matrix; return rep; }

        rhs_.resize(m);
        rates_.resize(m);

        constexpr R eps = std::numeric_limits<R>::epsilon();
        R kappa = R(ctl.tolerance);
        if (meas.rtol > R(0)) kappa = std::max(kappa, R(10) * eps / meas.rtol);

        R previous = 0;
        for (std::size_t k = 0; k < ctl.max_iterations; ++k) {
            for (std::size_t i = 0; i < s; ++i) {          // stage rates
                for (std::size_t a = 0; a < n_; ++a) ywork_[a] = y[a] + Z[i * n_ + a];
                f(t + tab.c[i] * hs, std::span<const R>(ywork_),
                  std::span<R>(rates_).subspan(i * n_, n_));
            }
            rep.nfev += s;

            for (std::size_t idx = 0; idx < m; ++idx) rhs_[idx] = -Z[idx];
            for (std::size_t i = 0; i < s; ++i)            // rhs = -G = -Z + hΣa_ij F_j
                for (std::size_t j = 0; j < s; ++j) {
                    const R w = hs * tab.A[i, j];
                    if (w == R(0)) continue;
                    being::axpy(w, std::span<const R>(rates_).subspan(j * n_, n_),
                                std::span<R>(rhs_).subspan(i * n_, n_));
                }
            lu->solve(rhs_);

            const R dn = meas.norm(rhs_, y);               // scaled ‖Δ‖, y cycled
            rep.iterations = k + 1;
            rep.increment  = dn;
            if (!std::isfinite(dn)) { rep.outcome = NewtonOutcome::nonfinite; return rep; }

            for (std::size_t idx = 0; idx < m; ++idx) Z[idx] += rhs_[idx];

            if (k == 0) {
                if (dn <= kappa * R(1e-2)) { rep.outcome = NewtonOutcome::converged; return rep; }
            } else {
                const R theta = dn / previous;
                rep.contraction = theta;
                if (!(theta < R(0.99))) { rep.outcome = NewtonOutcome::diverged; return rep; }
                const R eta = theta / (R(1) - theta);
                if (eta * dn <= kappa) { rep.outcome = NewtonOutcome::converged; return rep; }
                if (k + 1 < ctl.max_iterations) {          // hopeless-contraction forecast
                    const R left = R(ctl.max_iterations - 1 - k);
                    if (std::pow(theta, left) / (R(1) - theta) * dn > kappa) {
                        rep.outcome = NewtonOutcome::slow;
                        return rep;
                    }
                }
            }
            previous = dn;
        }
        rep.outcome = NewtonOutcome::max_iterations;
        return rep;
    }

private:
    std::size_t n_;
    essence::Matrix<R> J_;
    std::vector<R> ywork_, f0work_, fwork_, rhs_, rates_;
};

} // namespace irk::notion
