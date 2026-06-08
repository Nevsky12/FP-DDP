#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · being.hpp                                                    [layer 0]
//
//  BEING (Sein).  The immediate: number, magnitude, extent — and Measure
//  (Maß), Hegel's unity of quality and quantity, here a tolerance that turns
//  raw magnitude into significance.  Nothing is presupposed: this header
//  admits only the C++ standard library.
//
//  Exports:   Real, axpy, Interval, Measure.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>

namespace irk::being {

// Quality: what it is to be a number of the continuum.
template <typename R>
concept Real = std::floating_point<R>;

// Quantity: the one elementary act of linear combination,  y ← y + a·x.
template <Real R>
constexpr void axpy(R a, std::span<const R> x, std::span<R> y) noexcept {
    for (std::size_t i = 0; i < y.size(); ++i) y[i] += a * x[i];
}

// Extent: a directed segment of the independent variable.
template <Real R>
struct Interval {
    R t0{};
    R t1{};

    [[nodiscard]] constexpr R length() const noexcept { return std::abs(t1 - t0); }
    [[nodiscard]] constexpr R direction() const noexcept { return t1 >= t0 ? R(1) : R(-1); }
};

// Measure: absolute + relative tolerance inducing a scaled RMS norm.
// A vector is "small" only relative to what it measures.
template <Real R>
struct Measure {
    R rtol{R(1e-6)};
    R atol{R(1e-9)};

    [[nodiscard]] constexpr R scale(R reference) const noexcept {
        return atol + rtol * std::abs(reference);
    }

    // ‖v‖ with per-component scales taken from `ref`; if ref is shorter than v
    // it is cycled (block structure: v = s stacked copies of an n-vector space).
    [[nodiscard]] R norm(std::span<const R> v, std::span<const R> ref) const noexcept {
        if (v.empty() || ref.empty()) return R(0);
        R acc = 0;
        for (std::size_t i = 0; i < v.size(); ++i) {
            const R q = v[i] / scale(ref[i % ref.size()]);
            acc += q * q;
        }
        return std::sqrt(acc / R(v.size()));
    }

    // ‖v‖ scaled by max(|r1|, |r2|) componentwise — the norm used to judge a
    // step against both its departure and its arrival state.
    [[nodiscard]] R norm2ref(std::span<const R> v, std::span<const R> r1,
                             std::span<const R> r2) const noexcept {
        if (v.empty()) return R(0);
        R acc = 0;
        for (std::size_t i = 0; i < v.size(); ++i) {
            const R sc = atol + rtol * std::max(std::abs(r1[i]), std::abs(r2[i]));
            const R q  = v[i] / sc;
            acc += q * q;
        }
        return std::sqrt(acc / R(v.size()));
    }
};

} // namespace irk::being
