#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · notion/concepts.hpp                                          [layer 2]
//
//  NOTION I (Begriff): the universal.  Chelpanov defines per genus et
//  differentiam; so do we, literally, with C++ concepts.  A VectorField is
//  (genus) a mapping which (differentia) takes a time and a state and yields
//  a rate.  ProvidesJacobian is its species that also knows its own law of
//  variation; for the rest, the law is inferred by divided differences.
//
//  Exports:   VectorField, ProvidesJacobian, jacobian.
//  Includes:  being, essence/algebra.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>

#include "irk/being.hpp"
#include "irk/essence/algebra.hpp"

namespace irk::notion {

template <typename F, typename R>
concept VectorField =
    being::Real<R> &&
    requires(const F f, R t, std::span<const R> y, std::span<R> dy) {
        { f(t, y, dy) } -> std::same_as<void>;
    };

template <typename F, typename R>
concept ProvidesJacobian =
    VectorField<F, R> &&
    requires(const F f, R t, std::span<const R> y, essence::Matrix<R>& J) {
        { f.jacobian(t, y, J) } -> std::same_as<void>;
    };

// Fill J = ∂f/∂y at (t, y).  Analytic if the field provides it (J arrives
// zeroed; the field must set every nonzero entry), otherwise one-sided
// divided differences with the RADAU increment δ_j = √(ε·max(10⁻⁵, |y_j|)).
// Returns the number of field evaluations consumed.
template <being::Real R, typename F>
    requires VectorField<F, R>
std::size_t jacobian(const F& f, R t, std::span<const R> y, essence::Matrix<R>& J,
                     std::span<R> ywork, std::span<R> f0work, std::span<R> fwork) {
    const std::size_t n = y.size();
    J.set_zero();
    if constexpr (ProvidesJacobian<F, R>) {
        f.jacobian(t, y, J);
        return 0;
    } else {
        constexpr R eps = std::numeric_limits<R>::epsilon();
        f(t, y, f0work);
        for (std::size_t i = 0; i < n; ++i) ywork[i] = y[i];
        for (std::size_t j = 0; j < n; ++j) {
            const R saved = ywork[j];
            R delta = std::sqrt(eps * std::max(R(1e-5), std::abs(saved)));
            ywork[j] = saved + delta;
            delta    = ywork[j] - saved;          // exactly representable step
            f(t, std::span<const R>(ywork), fwork);
            const R inv = R(1) / delta;
            for (std::size_t i = 0; i < n; ++i) J[i, j] = (fwork[i] - f0work[i]) * inv;
            ywork[j] = saved;
        }
        return n + 1;
    }
}

} // namespace irk::notion
