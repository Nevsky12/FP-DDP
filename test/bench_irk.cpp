// ─────────────────────────────────────────────────────────────────────────────
//  test/bench_irk.cpp — fp-ddp side of the IRK performance/robustness bench.
//  One GL3 (s=3) interval of length dt WITH forward sensitivities per case,
//  via the hardened IrkStepper; case grids MUST match
//  compare/bench_irk_acados.py (joined on (model, case index)).
//
//  Per case: ok flag, leaves taken (>1 ⇒ substep splitting engaged), median
//  COLD time (warm-start state dropped per call) and median WARM time
//  (repeated call, the DDP relinearization pattern), x⁺.
//  All Jacobians are analytic (FD-validated at startup) so the comparison
//  against acados' codegen'd derivatives measures the integrators.
//  Output: logs/bench_irk_fpddp.txt
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "fpddp/irk_stepper.h"
#include "fpddp/fd_jacobian.h"

using namespace fpddp;

struct PendDyn {
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = -std::sin(x[0]) + u[0]; }
    void fx(double, CSpan x, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=-std::cos(x[0]); A(1,1)=0; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct VdpDyn {
    double mu = 5.0;
    void rhs(double, CSpan x, CSpan u, Span dx) const {
        dx[0] = x[1]; dx[1] = mu*(1.0 - x[0]*x[0])*x[1] - x[0] + u[0];
    }
    void fx(double, CSpan x, CSpan, Mat& A) const {
        A(0,0)=0; A(0,1)=1; A(1,0)=-2.0*mu*x[0]*x[1]-1.0; A(1,1)=mu*(1.0-x[0]*x[0]);
    }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct CartPoleDyn {   // analytic Jacobians (FD-validated in main)
    static constexpr double M = 1.0, m = 0.1, g = 9.81, l = 0.8;
    void rhs(double, CSpan x, CSpan u, Span dx) const {
        const double th = x[1], om = x[3], F = u[0];
        const double ct = std::cos(th), st = std::sin(th);
        const double den = M + m - m*ct*ct;
        dx[0] = x[2]; dx[1] = om;
        dx[2] = (-m*l*st*om*om + m*g*ct*st + F)/den;
        dx[3] = (-m*l*ct*st*om*om + F*ct + (M+m)*g*st)/(l*den);
    }
    void fx(double, CSpan x, CSpan u, Mat& A) const {
        const double th = x[1], om = x[3], F = u[0];
        const double ct = std::cos(th), st = std::sin(th);
        const double c2 = ct*ct - st*st;                       // cos 2θ
        const double den = M + m - m*ct*ct, dden = 2.0*m*ct*st;
        const double n2 = -m*l*st*om*om + m*g*ct*st + F;
        const double dn2_th = -m*l*ct*om*om + m*g*c2;
        const double n3 = -m*l*ct*st*om*om + F*ct + (M+m)*g*st;
        const double dn3_th = -m*l*c2*om*om - F*st + (M+m)*g*ct;
        A(0,2) = 1; A(1,3) = 1;
        A(2,1) = (dn2_th*den - n2*dden)/(den*den);
        A(2,3) = -2.0*m*l*st*om/den;
        A(3,1) = (dn3_th*den - n3*dden)/(l*den*den);
        A(3,3) = -2.0*m*l*ct*st*om/(l*den);
    }
    void fu(double, CSpan x, CSpan, Mat& B) const {
        const double th = x[1];
        const double ct = std::cos(th);
        const double den = M + m - m*ct*ct;
        B(2,0) = 1.0/den;
        B(3,0) = ct/(l*den);
    }
};

struct Case { std::string model; int idx; double dt; std::vector<double> x, u; };

static std::vector<Case> cases() {
    std::vector<Case> cs;
    int k = 0;
    for (double th : {0.5, 1.5, 3.0})
        for (double om : {0.0, 2.0, 8.0, 20.0})
            for (double uu : {0.0, 2.0})
                for (double dt : {0.05, 0.2, 0.5})
                    cs.push_back({"pendulum", k++, dt, {th, om}, {uu}});
    k = 0;
    for (double th : {0.1, 1.5, 3.0})
        for (double om : {0.0, 5.0, 15.0, 40.0})
            for (double v : {0.0, 5.0})
                for (double F : {0.0, 20.0})
                    for (double dt : {0.04, 0.2})
                        cs.push_back({"cartpole", k++, dt, {0.0, th, v, om}, {F}});
    k = 0;
    for (double x1 : {0.5, 2.0, 4.0})
        for (double x2 : {0.0, 3.0, 8.0})
            for (double dt : {0.1, 0.5})
                cs.push_back({"vdp", k++, dt, {x1, x2}, {0.0}});
    return cs;
}

template <class F> static double med_ns(F&& f, int reps) {
    std::vector<double> t(reps);
    for (int i = 0; i < 3; ++i) f();
    for (int i = 0; i < reps; ++i) {
        auto a = std::chrono::steady_clock::now();
        f();
        auto b = std::chrono::steady_clock::now();
        t[i] = std::chrono::duration<double, std::nano>(b - a).count();
    }
    std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
    return t[reps / 2];
}

// FD-validate the hand Jacobians of a model at a probe point
template <class D>
static bool check_jac(const D& d, std::vector<double> x, std::vector<double> u, const char* name) {
    const std::size_t n = x.size(), m = u.size();
    Mat A(n, n), Afd(n, n), B(n, m), Bfd(n, m);
    d.fx(0.0, x, u, A);
    d.fu(0.0, x, u, B);
    auto rhs = [&](double t, CSpan xx, CSpan uu, Span dd) { d.rhs(t, xx, uu, dd); };
    fd_state_jac(rhs, 0.0, x, u, Afd);
    fd_control_jac(rhs, 0.0, x, u, Bfd);
    double worst = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) worst = std::max(worst, std::abs(A(i,j) - Afd(i,j)));
        for (std::size_t j = 0; j < m; ++j) worst = std::max(worst, std::abs(B(i,j) - Bfd(i,j)));
    }
    if (worst > 1e-5) { std::printf("JACOBIAN MISMATCH %s: %.2e\n", name, worst); return false; }
    return true;
}

int main() {
    if (!check_jac(CartPoleDyn{}, {0.3, 1.2, -0.7, 5.0}, {3.0}, "cartpole") ||
        !check_jac(VdpDyn{5.0},   {1.3, -2.1},           {0.4}, "vdp")      ||
        !check_jac(PendDyn{},     {1.1, 0.7},            {0.2}, "pendulum")) return 1;

    AnyDynamics pend{PendDyn{}}, cart{CartPoleDyn{}}, vdp{VdpDyn{5.0}};
    auto dyn = [&](const std::string& m) -> const AnyDynamics& {
        return m == "pendulum" ? pend : m == "cartpole" ? cart : vdp;
    };
    IrkStepper st2(2, 1), st4(4, 1);
    auto stepper = [&](const std::string& m) -> IrkStepper& { return m == "cartpole" ? st4 : st2; };

    FILE* f = std::fopen("logs/bench_irk_fpddp.txt", "w");
    if (!f) { std::printf("cannot open logs/bench_irk_fpddp.txt\n"); return 1; }
    std::fprintf(f, "solver fp-ddp\nbench irk\n");

    const int reps = 200;
    int nfail = 0, nsplit = 0;
    for (const auto& c : cases()) {
        const auto& d = dyn(c.model);
        auto& s = stepper(c.model);
        std::vector<double> xn;
        Mat A, B;
        // cold: warm-start state dropped each call (pure function of the case)
        double t_cold = med_ns([&] { s.forget(); s.step(d, c.x, c.u, 0.0, c.dt, 1, xn, A, B); }, reps);
        // warm: repeated call on the same data (DDP relinearization pattern)
        s.forget(); s.step(d, c.x, c.u, 0.0, c.dt, 1, xn, A, B);
        double t_warm = med_ns([&] { s.step(d, c.x, c.u, 0.0, c.dt, 1, xn, A, B); }, reps);

        s.forget();
        const bool ok = s.step(d, c.x, c.u, 0.0, c.dt, 1, xn, A, B);
        const auto leaves = s.stats().leaves;
        if (!ok) ++nfail;
        if (ok && leaves > 1) ++nsplit;

        std::fprintf(f, "case %s %d %.6f %d %zu %.0f %.0f", c.model.c_str(), c.idx, c.dt,
                     ok ? 1 : 0, leaves, t_cold, t_warm);
        for (double v : xn) std::fprintf(f, " %.14e", v);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    std::printf("bench_irk: %zu cases, %d flagged failures, %d solved via substep splitting\n",
                cases().size(), nfail, nsplit);
    return 0;
}
