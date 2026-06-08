// ─────────────────────────────────────────────────────────────────────────────
//  irk · examples/demo.cpp
//
//  Five demonstrations, each a small experiment with a PASS/FAIL verdict:
//    1  Genesis of the tableau   — generated coefficients vs. the literature
//    2  Orders made visible      — fixed-step convergence slopes
//    3  A stiff scalar           — Prothero–Robinson with λ = -10⁶
//    4  Van der Pol, μ = 1000    — adaptive step & order on a relaxation cycle
//    5  Kepler, e = 0.6          — symplectic Gauss vs. dissipative Radau
//
//  Build:  g++ -std=c++23 -O2 -Wall -Wextra -Iinclude examples/demo.cpp -o demo
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "irk/irk.hpp"

namespace {

int g_failures = 0;

template <class... A>
void println(std::format_string<A...> fmt, A&&... args) {
    std::cout << std::format(fmt, std::forward<A>(args)...) << '\n';
}

void check(bool ok, std::string label) {
    println("  [{}] {}", ok ? "PASS" : "FAIL", label);
    if (!ok) ++g_failures;
}

void section(int k, std::string_view title) {
    println("\n{}", std::string(76, '-'));
    println(" {}  {}", k, title);
    println("{}", std::string(76, '-'));
}

// ── the cast of problems ────────────────────────────────────────────────────

struct Logistic {                            // y' = y(1-y), exact 1/(1+9e^{-t})
    void operator()(double, std::span<const double> y, std::span<double> dy) const {
        dy[0] = y[0] * (1.0 - y[0]);
    }
};

struct ProtheroRobinson {                    // y' = λ(y - sin t) + cos t, exact sin t
    double lambda;
    void operator()(double t, std::span<const double> y, std::span<double> dy) const {
        dy[0] = lambda * (y[0] - std::sin(t)) + std::cos(t);
    }
    void jacobian(double, std::span<const double>, irk::Matrix<double>& J) const {
        J(0, 0) = lambda;
    }
};

struct VanDerPol {
    double mu;
    void operator()(double, std::span<const double> y, std::span<double> dy) const {
        dy[0] = y[1];
        dy[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    }
    void jacobian(double, std::span<const double> y, irk::Matrix<double>& J) const {
        J(0, 1) = 1.0;
        J(1, 0) = -2.0 * mu * y[0] * y[1] - 1.0;
        J(1, 1) = mu * (1.0 - y[0] * y[0]);
    }
};

struct Kepler {                              // y = (q1, q2, p1, p2),  H = ½|p|² - 1/r
    void operator()(double, std::span<const double> y, std::span<double> dy) const {
        const double r2 = y[0] * y[0] + y[1] * y[1];
        const double r3 = r2 * std::sqrt(r2);
        dy[0] = y[2];
        dy[1] = y[3];
        dy[2] = -y[0] / r3;
        dy[3] = -y[1] / r3;
    }
    void jacobian(double, std::span<const double> y, irk::Matrix<double>& J) const {
        const double r2 = y[0] * y[0] + y[1] * y[1];
        const double r  = std::sqrt(r2);
        const double r3 = r2 * r, r5 = r3 * r2;
        J(0, 2) = 1.0;
        J(1, 3) = 1.0;
        J(2, 0) = -1.0 / r3 + 3.0 * y[0] * y[0] / r5;
        J(2, 1) = 3.0 * y[0] * y[1] / r5;
        J(3, 0) = 3.0 * y[0] * y[1] / r5;
        J(3, 1) = -1.0 / r3 + 3.0 * y[1] * y[1] / r5;
    }
    static double energy(std::span<const double> y) {
        const double r = std::sqrt(y[0] * y[0] + y[1] * y[1]);
        return 0.5 * (y[2] * y[2] + y[3] * y[3]) - 1.0 / r;
    }
};

// ── 1 · genesis of the tableau ──────────────────────────────────────────────

void demo_tableau() {
    section(1, "Genesis of the tableau: generated coefficients vs. the literature");

    {   // Radau IIA, s = 3 (order 5): the RADAU5 coefficients
        const double r6 = std::sqrt(6.0);
        const double c_lit[3] = {(4.0 - r6) / 10.0, (4.0 + r6) / 10.0, 1.0};
        const double b_lit[3] = {(16.0 - r6) / 36.0, (16.0 + r6) / 36.0, 1.0 / 9.0};
        const double A_lit[3][3] = {
            {(88.0 - 7.0 * r6) / 360.0, (296.0 - 169.0 * r6) / 1800.0, (-2.0 + 3.0 * r6) / 225.0},
            {(296.0 + 169.0 * r6) / 1800.0, (88.0 + 7.0 * r6) / 360.0, (-2.0 - 3.0 * r6) / 225.0},
            {(16.0 - r6) / 36.0, (16.0 + r6) / 36.0, 1.0 / 9.0}};

        const auto tab = irk::ButcherTableau<double>::make(irk::Family::radau_iia, 3);
        double dev = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            dev = std::max(dev, std::abs(tab.c[i] - c_lit[i]));
            dev = std::max(dev, std::abs(tab.b[i] - b_lit[i]));
            for (std::size_t j = 0; j < 3; ++j)
                dev = std::max(dev, std::abs(tab.A(i, j) - A_lit[i][j]));
        }
        println("  Radau IIA, s = 3 (order {}):  c = {{{:.15f}, {:.15f}, {:.15f}}}",
                tab.order, tab.c[0], tab.c[1], tab.c[2]);
        check(dev < 1e-13, std::format("Radau IIA s=3 matches RADAU5 constants   "
                                       "(max deviation {:.1e})", dev));
    }

    {   // Gauss, s = 2 (order 4): the Hammer–Hollingsworth coefficients
        const double r3 = std::sqrt(3.0);
        const double c_lit[2] = {0.5 - r3 / 6.0, 0.5 + r3 / 6.0};
        const double b_lit[2] = {0.5, 0.5};
        const double A_lit[2][2] = {{0.25, 0.25 - r3 / 6.0}, {0.25 + r3 / 6.0, 0.25}};

        const auto tab = irk::ButcherTableau<double>::make(irk::Family::gauss_legendre, 2);
        double dev = 0;
        for (std::size_t i = 0; i < 2; ++i) {
            dev = std::max(dev, std::abs(tab.c[i] - c_lit[i]));
            dev = std::max(dev, std::abs(tab.b[i] - b_lit[i]));
            for (std::size_t j = 0; j < 2; ++j)
                dev = std::max(dev, std::abs(tab.A(i, j) - A_lit[i][j]));
        }
        check(dev < 1e-13, std::format("Gauss s=2 matches Hammer–Hollingsworth   "
                                       "(max deviation {:.1e})", dev));
    }

    println("\n  order-condition residuals  max over B(s), C(s) moment identities:");
    println("    s    Gauss        Radau IIA");
    double worst = 0;
    for (std::size_t s = 2; s <= 7; ++s) {
        const auto g = irk::ButcherTableau<double>::make(irk::Family::gauss_legendre, s);
        const auto r = irk::ButcherTableau<double>::make(irk::Family::radau_iia, s);
        const double rg = g.order_condition_residual(), rr = r.order_condition_residual();
        worst = std::max({worst, rg, rr});
        println("    {}    {:.2e}     {:.2e}", s, rg, rr);
    }
    check(worst < 1e-10, std::format("all residuals below 1e-10 in double       "
                                     "(worst {:.1e})", worst));
}

// ── 2 · orders made visible ─────────────────────────────────────────────────

void demo_convergence() {
    section(2, "Orders made visible: fixed-step error slopes on the logistic equation");

    const std::vector<double> y0{0.1};
    const double t_end = 2.0;
    const double exact = 1.0 / (1.0 + 9.0 * std::exp(-t_end));

    auto error_at = [&](irk::Family fam, std::size_t s, double h) -> double {
        irk::Options<double> opt;
        opt.family       = fam;
        opt.fixed_step   = h;
        opt.fixed_stages = s;
        const auto res = irk::integrate(Logistic{}, irk::Interval<double>{0.0, t_end},
                                        std::span<const double>(y0),
                                        irk::Measure<double>{1e-12, 1e-12}, opt);
        if (!res) {
            check(false, std::format("fixed-step run broke down: {}", irk::name(res.error().kind)));
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::abs(res->y_end()[0] - exact);
    };

    println("  family      s  order   err(h)       err(h/2)     slope   expected");
    struct Case { irk::Family fam; std::size_t s; double h; };
    const Case cases[] = {
        {irk::Family::gauss_legendre, 2, 0.20}, {irk::Family::gauss_legendre, 3, 0.25},
        {irk::Family::gauss_legendre, 4, 0.40}, {irk::Family::radau_iia, 2, 0.10},
        {irk::Family::radau_iia, 3, 0.20},      {irk::Family::radau_iia, 4, 0.30},
    };
    for (const auto& [fam, s, h] : cases) {
        const double e1 = error_at(fam, s, h);
        const double e2 = error_at(fam, s, h / 2.0);
        const double slope = std::log2(e1 / e2);
        const unsigned expected = (fam == irk::Family::gauss_legendre) ? unsigned(2 * s)
                                                                       : unsigned(2 * s - 1);
        println("  {:<9}   {}  {:>4}    {:.3e}    {:.3e}    {:>5.2f}   {}",
                irk::name(fam), s, expected, e1, e2, slope, expected);
        const bool floored = e2 < 1e-12;
        check(std::abs(slope - double(expected)) <= 0.5 || floored,
              std::format("{} s={} slope {:.2f} ~ {}{}", irk::name(fam), s, slope, expected,
                          floored ? "  (at rounding floor)" : ""));
    }
}

// ── 3 · a stiff scalar ──────────────────────────────────────────────────────

void demo_prothero_robinson() {
    section(3, "A stiff scalar: Prothero-Robinson, lambda = -1e6, t in [0, 10]");

    const ProtheroRobinson f{-1e6};
    const std::vector<double> y0{0.0};
    const auto res = irk::integrate(f, irk::Interval<double>{0.0, 10.0},
                                    std::span<const double>(y0),
                                    irk::Measure<double>{1e-8, 1e-10});
    if (!res) {
        check(false, std::format("integration broke down: {}", irk::name(res.error().kind)));
        return;
    }
    const auto& sol = *res;
    double hmax = 0;
    for (const auto& nd : sol.nodes()) hmax = std::max(hmax, std::abs(nd.hs));
    const double err = std::abs(sol.y_end()[0] - std::sin(10.0));

    println("  steps accepted     {}", sol.stats().naccept);
    println("  largest step       {:.3f}   ({:.0e} x the stiff time scale 1/|lambda|)",
            hmax, hmax * 1e6);
    println("  error at t = 10    {:.3e}   (rtol = 1e-8)", err);
    check(err < 1e-5, std::format("global error {:.1e} < 1e-5", err));
    check(sol.stats().naccept < 1000,
          std::format("stiffness defeated in {} steps", sol.stats().naccept));

    double node_err = 0;                     // superconvergence at the nodes …
    for (const auto& nd : sol.nodes())
        node_err = std::max(node_err, std::abs(sol.eval(nd.t)[0] - std::sin(nd.t)));
    double dense_err = 0;                    // … stage-order accuracy between them
    for (int k = 1; k < 40; ++k) {
        const double t = 10.0 * double(k) / 40.0;
        dense_err = std::max(dense_err, std::abs(sol.eval(t)[0] - std::sin(t)));
    }
    println("  dense output       {:.3e} at the nodes (superconvergent),", node_err);
    println("                     {:.3e} between them (stage order, h ~ 3)", dense_err);
    check(node_err < 1e-6 && dense_err < 1e-3,       // ~rtol at nodes, h^{s+1} between
          std::format("dense output: {:.0e} at nodes / {:.0e} between", node_err, dense_err));
}

// ── 4 · Van der Pol ─────────────────────────────────────────────────────────

void demo_van_der_pol() {
    section(4, "Van der Pol, mu = 1000, t in [0, 3000]: adaptive step and order");

    const VanDerPol f{1000.0};
    const std::vector<double> y0{2.0, 0.0};
    const irk::Interval<double> iv{0.0, 3000.0};

    const auto run = irk::integrate(f, iv, std::span<const double>(y0),
                                    irk::Measure<double>{1e-8, 1e-8});
    const auto ref = irk::integrate(f, iv, std::span<const double>(y0),
                                    irk::Measure<double>{1e-11, 1e-11});
    if (!run || !ref) {
        check(false, "a Van der Pol run broke down");
        return;
    }
    const auto& st = run->stats();
    println("  steps accepted     {}    rejected: {} (error) + {} (Newton)",
            st.naccept, st.nreject_error, st.nreject_newton);
    println("  f evaluations      {}    Jacobians: {}    LU factorizations: {}",
            st.nfev, st.njac, st.nlu);
    println("  Newton iterations  {}    ({:.2f} per attempt)", st.niter,
            double(st.niter) / double(st.naccept + st.nreject_error + st.nreject_newton));
    println("  accepted steps by stage count (the order ladder at work):");
    for (const auto& [s, count] : st.accepted_by_stage)
        println("    s = {}  (order {:>2}):  {:>6} steps", s, 2 * s - 1, count);

    double diff = 0;
    for (std::size_t i = 0; i < 2; ++i)
        diff = std::max(diff, std::abs(run->y_end()[i] - ref->y_end()[i]) /
                                  (1.0 + std::abs(ref->y_end()[i])));
    println("  y(3000)            ({:+.9f}, {:+.9f})", run->y_end()[0], run->y_end()[1]);
    check(diff < 1e-5, std::format("agrees with rtol=1e-11 reference to {:.1e}", diff));
}

// ── 5 · Kepler ──────────────────────────────────────────────────────────────

void demo_kepler() {
    section(5, "Kepler, e = 0.6, 50 periods: symplectic Gauss vs. dissipative Radau");

    const std::vector<double> y0{0.4, 0.0, 0.0, 2.0};   // perihelion of an e = 0.6 orbit
    const double H0 = Kepler::energy(y0);               // = -1/2
    const irk::Interval<double> iv{0.0, 100.0 * std::numbers::pi};
    const irk::Measure<double> tight{1e-12, 1e-12};

    auto drift_of = [&](irk::Family fam) -> std::pair<double, std::size_t> {
        irk::Options<double> opt;
        opt.family       = fam;
        opt.fixed_step   = 0.05;
        opt.fixed_stages = 3;
        const auto res = irk::integrate(Kepler{}, iv, std::span<const double>(y0), tight, opt);
        if (!res) {
            check(false, std::format("{} Kepler run broke down: {}", irk::name(fam),
                                     irk::name(res.error().kind)));
            return {std::numeric_limits<double>::quiet_NaN(), 0};
        }
        double drift = 0;
        for (const auto& nd : res->nodes())
            drift = std::max(drift, std::abs(Kepler::energy(nd.y) - H0));
        drift = std::max(drift, std::abs(Kepler::energy(res->y_end()) - H0));
        return {drift, res->steps()};
    };

    const auto [gauss_drift, gsteps] = drift_of(irk::Family::gauss_legendre);
    const auto [radau_drift, rsteps] = drift_of(irk::Family::radau_iia);

    println("  fixed h = 0.05, s = 3, {} steps, H0 = {:.1f}", gsteps, H0);
    println("  max |H - H0|   Gauss s=3 (order 6, symplectic):     {:.3e}", gauss_drift);
    println("  max |H - H0|   Radau IIA s=3 (order 5, damped):     {:.3e}", radau_drift);
    check(gauss_drift < 1e-7,
          std::format("Gauss energy drift {:.1e} stays bounded", gauss_drift));
    check(gauss_drift * 10.0 < radau_drift,
          std::format("Gauss beats Radau on |dH| by {:.0f}x", radau_drift / gauss_drift));
}

} // namespace

int main() {
    println("irk: a variable-order implicit Runge-Kutta integrator   (C++23)");
    demo_tableau();
    demo_convergence();
    demo_prothero_robinson();
    demo_van_der_pol();
    demo_kepler();

    println("\n{}", std::string(76, '='));
    if (g_failures == 0) println(" all checks passed");
    else                 println(" {} CHECK(S) FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
}
