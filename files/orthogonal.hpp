#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · essence/orthogonal.hpp                                       [layer 1]
//
//  ESSENCE II (Wesen): appearance.  The Legendre family is the inner law from
//  which every node placement appears; Gauss and Radau IIA points are not
//  invented but *found* — roots of one recurrence, two species of one genus.
//
//  Exports:   legendre, gauss_nodes, radau_iia_nodes.
//  Includes:  being.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "irk/being.hpp"

namespace irk::essence {

// P_n and P_n' on [-1, 1] by the three-term recurrence and the stable
// derivative recurrence  P'_{k+1} = P'_{k-1} + (2k+1)·P_k.
template <being::Real R>
[[nodiscard]] constexpr std::pair<R, R> legendre(unsigned n, R x) noexcept {
    if (n == 0) return {R(1), R(0)};
    R pm1 = R(1), p = x;     // P_0, P_1
    R dm1 = R(0), d = R(1);  // P_0', P_1'
    for (unsigned k = 1; k < n; ++k) {
        const R pn = (R(2 * k + 1) * x * p - R(k) * pm1) / R(k + 1);
        const R dn = dm1 + R(2 * k + 1) * p;
        pm1 = p;  p = pn;
        dm1 = d;  d = dn;
    }
    return {p, d};
}

// Compile-time proofs that the recurrence is exact where exactness is binary.
static_assert(legendre<double>(0, 0.25).first == 1.0);
static_assert(legendre<double>(2, 0.0).first == -0.5);   // P_2(0) = -1/2
static_assert(legendre<double>(4, 1.0).first == 1.0);    // P_n(1) = 1
static_assert(legendre<double>(3, 1.0).second == 6.0);   // P_3'(1) = n(n+1)/2

namespace detail {

// All simple roots of fn (returning {value, slope}) inside (lo, hi):
// uniform sign scan, bisection to machine width, one or two Newton polishes.
template <being::Real R, typename Fn>
[[nodiscard]] std::vector<R> bracketed_roots(Fn&& fn, R lo, R hi, std::size_t count) {
    constexpr R eps = std::numeric_limits<R>::epsilon();
    const std::size_t cells = std::max<std::size_t>(256, 96 * std::max<std::size_t>(count, 1));

    std::vector<R> roots;
    roots.reserve(count);

    R xp = lo;
    R fp = fn(xp).first;
    for (std::size_t i = 1; i <= cells; ++i) {
        const R x = lo + (hi - lo) * R(i) / R(cells);
        const R f = fn(x).first;
        if ((fp < R(0)) != (f < R(0))) {
            R a = xp, b = x, fa = fp;
            for (int it = 0; it < 200 && (b - a) > eps * (std::abs(a) + std::abs(b) + R(1)); ++it) {
                const R m  = a + (b - a) / R(2);
                const R fm = fn(m).first;
                if (fm == R(0)) { a = b = m; break; }
                if ((fm < R(0)) == (fa < R(0))) { a = m; fa = fm; } else { b = m; }
            }
            R r = a + (b - a) / R(2);
            for (int polish = 0; polish < 2; ++polish) {       // Newton, kept in bracket
                const auto [v, s] = fn(r);
                if (s == R(0)) break;
                const R nx = r - v / s;
                if (nx >= a && nx <= b) r = nx;
            }
            roots.push_back(r);
        }
        xp = x;
        fp = f;
    }
    if (roots.size() != count)
        throw std::logic_error("irk: root bracketing found an unexpected count");
    return roots;
}

} // namespace detail

// Gauss–Legendre collocation points on (0, 1): roots of P_s, mapped by (x+1)/2.
template <being::Real R>
[[nodiscard]] std::vector<R> gauss_nodes(std::size_t s) {
    auto fn = [s](R x) { return legendre<R>(static_cast<unsigned>(s), x); };
    auto xs = detail::bracketed_roots<R>(fn, R(-1), R(1), s);
    for (auto& x : xs) x = (x + R(1)) / R(2);
    return xs;
}

// Radau IIA points on (0, 1]: the s-1 interior roots of
//   φ(x) = (P_s(x) - P_{s-1}(x)) / (x - 1)
// on (-1, 1), mapped by (x+1)/2, with the right endpoint c = 1 appended.
template <being::Real R>
[[nodiscard]] std::vector<R> radau_iia_nodes(std::size_t s) {
    std::vector<R> c;
    if (s > 1) {
        auto fn = [s](R x) -> std::pair<R, R> {
            const auto [ps, dps] = legendre<R>(static_cast<unsigned>(s), x);
            const auto [pm, dpm] = legendre<R>(static_cast<unsigned>(s - 1), x);
            const R psi  = ps - pm;
            const R dpsi = dps - dpm;
            const R den  = x - R(1);
            return {psi / den, (dpsi * den - psi) / (den * den)};
        };
        c = detail::bracketed_roots<R>(fn, R(-1), R(1) - R(1e-6), s - 1);
        for (auto& x : c) x = (x + R(1)) / R(2);
    }
    c.push_back(R(1));
    return c;
}

} // namespace irk::essence
