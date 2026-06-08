#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · essence/algebra.hpp                                          [layer 1]
//
//  ESSENCE I (Wesen): ground and relation.  Behind every state stands a web
//  of linear relations; the Matrix is relation made explicit, and the PLU
//  factorization is its ground — the decomposition through which any
//  consequence (a solve) is mediated.
//
//  Exports:   Matrix, Singular, PLU.
//  Includes:  being.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstddef>
#include <expected>
#include <span>
#include <utility>
#include <vector>

#include "irk/being.hpp"

namespace irk::essence {

template <being::Real R>
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), a_(rows * cols, R(0)) {}

    [[nodiscard]] static Matrix identity(std::size_t n) {
        Matrix m(n, n);
        for (std::size_t k = 0; k < n; ++k) m[k, k] = R(1);
        return m;
    }

    // C++23 multidimensional subscript.
    [[nodiscard]] constexpr R& operator[](std::size_t i, std::size_t j) noexcept {
        return a_[i * cols_ + j];
    }
    [[nodiscard]] constexpr const R& operator[](std::size_t i, std::size_t j) const noexcept {
        return a_[i * cols_ + j];
    }

    [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }

    [[nodiscard]] std::span<R> row(std::size_t i) noexcept {
        return std::span<R>(a_).subspan(i * cols_, cols_);
    }
    [[nodiscard]] std::span<const R> row(std::size_t i) const noexcept {
        return std::span<const R>(a_).subspan(i * cols_, cols_);
    }

    void set_zero() noexcept { for (auto& e : a_) e = R(0); }

private:
    std::size_t rows_{0}, cols_{0};
    std::vector<R> a_;
};

// The negative moment of factorization: a vanished pivot.
struct Singular {
    std::size_t pivot{};
};

// PLU factorization with partial pivoting (LAPACK-style ipiv).
template <being::Real R>
class PLU {
public:
    PLU() = default;

    [[nodiscard]] static std::expected<PLU, Singular> factor(Matrix<R> m) {
        const std::size_t n = m.rows();
        std::vector<std::size_t> piv(n);
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t p = k;
            R best = std::abs(m[k, k]);
            for (std::size_t r = k + 1; r < n; ++r)
                if (const R cand = std::abs(m[r, k]); cand > best) { best = cand; p = r; }
            if (!(best > R(0))) return std::unexpected(Singular{k});  // also rejects NaN
            piv[k] = p;
            if (p != k) {
                auto rk = m.row(k), rp = m.row(p);
                for (std::size_t c = 0; c < n; ++c) std::swap(rk[c], rp[c]);
            }
            const R inv = R(1) / m[k, k];
            for (std::size_t r = k + 1; r < n; ++r) {
                const R l = (m[r, k] *= inv);
                if (l == R(0)) continue;
                for (std::size_t c = k + 1; c < n; ++c) m[r, c] -= l * m[k, c];
            }
        }
        return PLU(std::move(m), std::move(piv));
    }

    // Solve LU·x = P·b in place.
    void solve(std::span<R> x) const noexcept {
        const std::size_t n = piv_.size();
        for (std::size_t k = 0; k < n; ++k)
            if (piv_[k] != k) std::swap(x[k], x[piv_[k]]);
        for (std::size_t r = 1; r < n; ++r) {            // forward, unit lower
            R acc = x[r];
            for (std::size_t c = 0; c < r; ++c) acc -= lu_[r, c] * x[c];
            x[r] = acc;
        }
        for (std::size_t rr = n; rr-- > 0;) {            // backward, upper
            R acc = x[rr];
            for (std::size_t c = rr + 1; c < n; ++c) acc -= lu_[rr, c] * x[c];
            x[rr] = acc / lu_[rr, rr];
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return piv_.size(); }

private:
    PLU(Matrix<R> lu, std::vector<std::size_t> piv) : lu_(std::move(lu)), piv_(std::move(piv)) {}

    Matrix<R> lu_;
    std::vector<std::size_t> piv_;
};

} // namespace irk::essence
