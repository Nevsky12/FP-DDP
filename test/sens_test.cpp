// ─────────────────────────────────────────────────────────────────────────────
//  test/sens_test.cpp — validate the inline IRK discrete sensitivities
//  A = ∂x_end/∂x_begin, B = ∂x_end/∂u against central finite differences of the
//  step map, with the discretization pinned (fixed step + stages) so the map is
//  smooth.  Linear field → exact; pendulum over several steps → exercises the
//  cross-step chaining.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

#include "irk/irk.hpp"

using R = double;

// ── controlled fields ───────────────────────────────────────────────────────
struct Linear {                       // ẋ = Ac x + Bc u
    std::vector<R> Ac, Bc, u;         // Ac n×n, Bc n×m, u m (row-major)
    std::size_t n, m;
    void operator()(R, std::span<const R> x, std::span<R> dx) const {
        for (std::size_t i = 0; i < n; ++i) {
            R a = 0;
            for (std::size_t j = 0; j < n; ++j) a += Ac[i * n + j] * x[j];
            for (std::size_t j = 0; j < m; ++j) a += Bc[i * m + j] * u[j];
            dx[i] = a;
        }
    }
    void jacobian(R, std::span<const R>, irk::Matrix<R>& J) const {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) J(i, j) = Ac[i * n + j];
    }
    void control_jacobian(R, std::span<const R>, irk::Matrix<R>& B) const {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < m; ++j) B(i, j) = Bc[i * m + j];
    }
    std::size_t control_dim() const { return m; }
};

struct Pendulum {                     // ẋ₁ = x₂,  ẋ₂ = −sin x₁ − d x₂ + u
    R d, u;
    void operator()(R, std::span<const R> x, std::span<R> dx) const {
        dx[0] = x[1];
        dx[1] = -std::sin(x[0]) - d * x[1] + u;
    }
    void jacobian(R, std::span<const R> x, irk::Matrix<R>& J) const {
        J(0, 0) = 0; J(0, 1) = 1; J(1, 0) = -std::cos(x[0]); J(1, 1) = -d;
    }
    void control_jacobian(R, std::span<const R>, irk::Matrix<R>& B) const { B(0, 0) = 0; B(1, 0) = 1; }
    std::size_t control_dim() const { return 1; }
};

// ── helpers ──────────────────────────────────────────────────────────────────
static irk::Options<R> opt_fixed(R h, std::size_t s) {
    irk::Options<R> o;
    o.family = irk::Family::gauss_legendre;
    o.fixed_step = h;
    o.fixed_stages = s;
    return o;
}

template <class F>
static std::vector<R> y_end(const F& f, std::span<const R> x0, R T, R h, std::size_t s) {
    auto sol = irk::integrate(f, irk::Interval<R>{0.0, T}, x0, irk::Measure<R>{1e-14, 1e-14}, opt_fixed(h, s));
    return std::vector<R>(sol->y_end().begin(), sol->y_end().end());
}

template <class F>
static irk::Matrix<R> fd_state(const F& f, std::span<const R> x0, R T, R h, std::size_t s,
                               std::size_t n, R eps) {
    irk::Matrix<R> A(n, n);
    for (std::size_t a = 0; a < n; ++a) {
        std::vector<R> xp(x0.begin(), x0.end()), xm(x0.begin(), x0.end());
        xp[a] += eps; xm[a] -= eps;
        auto yp = y_end(f, std::span<const R>(xp), T, h, s);
        auto ym = y_end(f, std::span<const R>(xm), T, h, s);
        for (std::size_t i = 0; i < n; ++i) A(i, a) = (yp[i] - ym[i]) / (2 * eps);
    }
    return A;
}

static double maxabs(const irk::Matrix<R>& X, const irk::Matrix<R>& Y) {
    double e = 0;
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) e = std::max(e, std::abs(X(i, j) - Y(i, j)));
    return e;
}

static int g_fail = 0;
static void check(bool ok, const char* what, double err) {
    std::printf("  [%s] %s (max |analytic - FD| = %.2e)\n", ok ? "PASS" : "FAIL", what, err);
    if (!ok) ++g_fail;
}

int main() {
    std::printf("irk sensitivity check (inline analytic vs finite difference)\n");
    const R T = 0.4, eps = 1e-6;

    // 1 · linear system, single step (h = T): variational sensitivity is exact
    {
        Linear f{{0.0, 1.0, -2.0, -0.3}, {0.0, 1.0}, {0.5}, 2, 1};
        std::vector<R> x0{0.7, -0.2};
        auto sol = irk::integrate(f, irk::Interval<R>{0.0, T}, std::span<const R>(x0),
                                  irk::Measure<R>{1e-14, 1e-14}, opt_fixed(T, 3));
        if (!sol) { std::printf("  [FAIL] linear integrate broke down\n"); return 1; }
        irk::Matrix<R> A = sol->state_sensitivity(), B = sol->control_sensitivity();

        irk::Matrix<R> Afd = fd_state(f, std::span<const R>(x0), T, T, 3, 2, eps);
        irk::Matrix<R> Bfd(2, 1);
        { Linear fp = f, fm = f; fp.u[0] += eps; fm.u[0] -= eps;
          auto yp = y_end(fp, std::span<const R>(x0), T, T, 3);
          auto ym = y_end(fm, std::span<const R>(x0), T, T, 3);
          for (std::size_t i = 0; i < 2; ++i) Bfd(i, 0) = (yp[i] - ym[i]) / (2 * eps); }

        check(maxabs(A, Afd) < 1e-7, "linear, 1 step : state   A", maxabs(A, Afd));
        check(maxabs(B, Bfd) < 1e-7, "linear, 1 step : control B", maxabs(B, Bfd));
    }

    // 2 · pendulum, multi-step (h = T/4): exercises cross-step chaining
    {
        Pendulum f{0.1, 0.3};
        std::vector<R> x0{0.5, 0.0};
        const R h = T / 4;
        auto sol = irk::integrate(f, irk::Interval<R>{0.0, T}, std::span<const R>(x0),
                                  irk::Measure<R>{1e-14, 1e-14}, opt_fixed(h, 3));
        if (!sol) { std::printf("  [FAIL] pendulum integrate broke down\n"); return 1; }
        std::printf("  (pendulum integrated in %zu internal steps)\n", sol->steps());
        irk::Matrix<R> A = sol->state_sensitivity(), B = sol->control_sensitivity();

        irk::Matrix<R> Afd = fd_state(f, std::span<const R>(x0), T, h, 3, 2, eps);
        irk::Matrix<R> Bfd(2, 1);
        { Pendulum fp = f, fm = f; fp.u += eps; fm.u -= eps;
          auto yp = y_end(fp, std::span<const R>(x0), T, h, 3);
          auto ym = y_end(fm, std::span<const R>(x0), T, h, 3);
          for (std::size_t i = 0; i < 2; ++i) Bfd(i, 0) = (yp[i] - ym[i]) / (2 * eps); }

        check(maxabs(A, Afd) < 1e-6, "pendulum, 4 steps : state   A (chaining)", maxabs(A, Afd));
        check(maxabs(B, Bfd) < 1e-6, "pendulum, 4 steps : control B (chaining)", maxabs(B, Bfd));
    }

    std::printf(g_fail == 0 ? "\n all sensitivity checks passed\n" : "\n %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
