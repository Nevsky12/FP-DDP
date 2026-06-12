// ─────────────────────────────────────────────────────────────────────────────
//  test/stress_fpddp.cpp — fp-ddp side of the acados stress comparison.
//  One binary, problem selected by argv[1]; problem definitions MUST match
//  compare/stress_acados.py exactly (dynamics, weights, N, dt, x0, u_ref).
//
//  All problems start from the acados default initialization (x_i ≡ x0, u ≡ 0,
//  dynamically infeasible) via the 3-arg solve, so the forced-full-step path is
//  exercised and both solvers iterate from the identical first point.
//  Settings = acados scripts: DDP + MeritBacktracking + adaptive LM,
//  tol_stat = tol_eq = 1e-6, max_iter = 300, IRK GL3 / 1 step.
//
//  Output: logs/stress_<name>_fpddp.txt with per-iteration stats + trajectory.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "fpddp/ddp.h"
#include "fpddp/fd_jacobian.h"

using namespace fpddp;

// generic diagonal quadratic cost: stage ½(xᵀdiag(q)x + (u−uref)ᵀdiag(ru)(u−uref))·dt,
// terminal ½ xᵀdiag(qf)x — identical to LINEAR_LS with W=diag([q,ru])·dt, yref=[0,uref].
struct QuadCostV {
    double dt;
    std::vector<double> q, ru, qf, uref;
    double stage(double, CSpan x, CSpan u) const {
        double c = 0;
        for (std::size_t i = 0; i < q.size(); ++i) c += q[i] * x[i] * x[i];
        for (std::size_t j = 0; j < ru.size(); ++j) { double d = u[j] - uref[j]; c += ru[j] * d * d; }
        return 0.5 * c * dt;
    }
    void stage_grad(double, CSpan x, CSpan u, Span lx, Span lu) const {
        for (std::size_t i = 0; i < q.size(); ++i) lx[i] = q[i] * x[i] * dt;
        for (std::size_t j = 0; j < ru.size(); ++j) lu[j] = ru[j] * (u[j] - uref[j]) * dt;
    }
    void stage_hess(double, CSpan, CSpan, Mat& Q, Mat& R, Mat&) const {
        for (std::size_t i = 0; i < q.size(); ++i) Q(i, i) = q[i] * dt;
        for (std::size_t j = 0; j < ru.size(); ++j) R(j, j) = ru[j] * dt;
    }
    double term(CSpan x) const {
        double c = 0;
        for (std::size_t i = 0; i < qf.size(); ++i) c += qf[i] * x[i] * x[i];
        return 0.5 * c;
    }
    void term_grad(CSpan x, Span lx) const { for (std::size_t i = 0; i < qf.size(); ++i) lx[i] = qf[i] * x[i]; }
    void term_hess(CSpan, Mat& Q) const { for (std::size_t i = 0; i < qf.size(); ++i) Q(i, i) = qf[i]; }
};

// ── dynamics ─────────────────────────────────────────────────────────────────
struct PendDyn {  // ẋ1 = x2, ẋ2 = −sin x1 + u
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = -std::sin(x[0]) + u[0]; }
    void fx(double, CSpan x, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=-std::cos(x[0]); A(1,1)=0; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct DIntDyn {  // double integrator (LQ, Newton-exact sanity)
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = u[0]; }
    void fx(double, CSpan, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=0; A(1,1)=0; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct VdpDyn {   // Van der Pol: ẋ1 = x2, ẋ2 = μ(1−x1²)x2 − x1 + u
    double mu = 1.0;
    void rhs(double, CSpan x, CSpan u, Span dx) const {
        dx[0] = x[1]; dx[1] = mu * (1.0 - x[0]*x[0]) * x[1] - x[0] + u[0];
    }
    void fx(double, CSpan x, CSpan, Mat& A) const {
        A(0,0)=0; A(0,1)=1;
        A(1,0) = -2.0*mu*x[0]*x[1] - 1.0; A(1,1) = mu*(1.0 - x[0]*x[0]);
    }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct UnicycleDyn {  // ẋ = v cosθ, ẏ = v sinθ, θ̇ = ω ; x=[px,py,θ], u=[v,ω]
    void rhs(double, CSpan x, CSpan u, Span dx) const {
        dx[0] = u[0]*std::cos(x[2]); dx[1] = u[0]*std::sin(x[2]); dx[2] = u[1];
    }
    void fx(double, CSpan x, CSpan u, Mat& A) const {
        A(0,2) = -u[0]*std::sin(x[2]); A(1,2) = u[0]*std::cos(x[2]);
    }
    void fu(double, CSpan x, CSpan, Mat& B) const {
        B(0,0)=std::cos(x[2]); B(1,0)=std::sin(x[2]); B(2,1)=1;
    }
};
struct CartPoleDyn {  // acados pendulum_ode model; θ=0 upright (unstable)
    static constexpr double M = 1.0, m = 0.1, g = 9.81, l = 0.8;
    void rhs(double, CSpan x, CSpan u, Span dx) const {
        const double th = x[1], om = x[3], F = u[0];
        const double ct = std::cos(th), st = std::sin(th);
        const double den = M + m - m*ct*ct;
        dx[0] = x[2];
        dx[1] = om;
        dx[2] = (-m*l*st*om*om + m*g*ct*st + F) / den;
        dx[3] = (-m*l*ct*st*om*om + F*ct + (M+m)*g*st) / (l*den);
    }
    void fx(double t, CSpan x, CSpan u, Mat& A) const {
        fd_state_jac([this](double tt, CSpan xx, CSpan uu, Span dd){ rhs(tt,xx,uu,dd); }, t, x, u, A);
    }
    void fu(double t, CSpan x, CSpan u, Mat& B) const {
        fd_control_jac([this](double tt, CSpan xx, CSpan uu, Span dd){ rhs(tt,xx,uu,dd); }, t, x, u, B);
    }
};
struct QuadrotorDyn {  // planar quadrotor: x=[px,pz,θ,vx,vz,ω], u=[f1,f2]
    static constexpr double m = 1.0, J = 0.05, larm = 0.2, g = 9.81;
    void rhs(double, CSpan x, CSpan u, Span dx) const {
        const double th = x[2], F = u[0] + u[1];
        dx[0] = x[3]; dx[1] = x[4]; dx[2] = x[5];
        dx[3] = -F*std::sin(th)/m;
        dx[4] =  F*std::cos(th)/m - g;
        dx[5] = larm*(u[0]-u[1])/J;
    }
    void fx(double, CSpan x, CSpan u, Mat& A) const {
        const double th = x[2], F = u[0] + u[1];
        A(0,3)=1; A(1,4)=1; A(2,5)=1;
        A(3,2) = -F*std::cos(th)/m;
        A(4,2) = -F*std::sin(th)/m;
    }
    void fu(double, CSpan x, CSpan, Mat& B) const {
        const double th = x[2];
        B(3,0)=-std::sin(th)/m; B(3,1)=-std::sin(th)/m;
        B(4,0)= std::cos(th)/m; B(4,1)= std::cos(th)/m;
        B(5,0)= larm/J;         B(5,1)=-larm/J;
    }
};

// ── problem registry (keep in sync with compare/stress_acados.py) ────────────
struct Problem {
    std::string name;
    int N, nx, nu;
    double dt;
    std::vector<double> x0, q, ru, qf, uref;
    AnyDynamics dyn;
};

static std::vector<Problem> registry() {
    const double uh = QuadrotorDyn::m * QuadrotorDyn::g / 2.0;   // hover thrust per rotor
    std::vector<Problem> ps;
    ps.push_back({"pendulum",      40, 2, 1, 0.05, {2.0, 0.0},               {1, 0.1},               {0.01},      {50, 50},                  {0},        AnyDynamics(PendDyn{})});
    ps.push_back({"dintegrator",   30, 2, 1, 0.05, {1.0, 0.0},               {1, 1},                 {0.1},       {5, 5},                    {0},        AnyDynamics(DIntDyn{})});
    ps.push_back({"vdp",           50, 2, 1, 0.10, {2.0, 0.0},               {1, 1},                 {0.1},       {10, 10},                  {0},        AnyDynamics(VdpDyn{})});
    ps.push_back({"unicycle",      40, 3, 2, 0.05, {1.0, 0.5, 0.8},          {1, 1, 0.5},            {0.1, 0.1},  {50, 50, 20},              {0, 0},     AnyDynamics(UnicycleDyn{})});
    ps.push_back({"cartpole",      30, 4, 1, 0.04, {0.25, 0.08, 0.0, 0.0},   {10, 10, 0.1, 0.1},     {0.01},      {100, 100, 10, 10},        {0},        AnyDynamics(CartPoleDyn{})});
    ps.push_back({"cartpole_long", 50, 4, 1, 0.04, {0.30, 0.15, 0.0, 0.0},   {10, 10, 0.1, 0.1},     {0.01},      {100, 100, 10, 10},        {0},        AnyDynamics(CartPoleDyn{})});
    ps.push_back({"quadrotor",     40, 6, 2, 0.05, {0.5, 0.5, 0.3, 0, 0, 0}, {10, 10, 10, 1, 1, 1},  {0.1, 0.1},  {100, 100, 100, 10, 10, 10}, {uh, uh}, AnyDynamics(QuadrotorDyn{})});
    // hard cases meant to force backtracking (alpha < 1) and LM growth
    ps.push_back({"pendulum_hard",  60, 2, 1, 0.05, {3.1, 0.0},              {1, 0.1},               {0.001},     {100, 100},                {0},        AnyDynamics(PendDyn{})});
    ps.push_back({"vdp_hard",       60, 2, 1, 0.10, {3.0, 1.0},              {1, 1},                 {0.01},      {10, 10},                  {0},        AnyDynamics(VdpDyn{3.0})});
    ps.push_back({"cartpole_swing", 50, 4, 1, 0.04, {0.0, 2.5, 0.0, 0.0},    {1, 5, 0.1, 0.1},       {0.01},      {100, 100, 10, 10},        {0},        AnyDynamics(CartPoleDyn{})});
    return ps;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: stress_fpddp <problem|all>\n"); return 2; }
    const std::string want = argv[1];
    int rc = 0;

    for (auto& p : registry()) {
        if (want != "all" && want != p.name) continue;

        std::vector<double> tg(p.N + 1);
        for (int i = 0; i <= p.N; ++i) tg[i] = i * p.dt;
        Ddp solver(p.dyn, AnyCost(QuadCostV{p.dt, p.q, p.ru, p.qf, p.uref}), p.N, p.nx, p.nu, tg);

        // acados default initialization: x_i ≡ x0, u ≡ 0 (dynamically infeasible)
        std::vector<std::vector<double>> xg(p.N + 1, p.x0), ug(p.N, std::vector<double>(p.nu, 0.0));

        DdpOptions opt;
        opt.tol_stat = 1e-6; opt.tol_eq = 1e-6; opt.max_iter = 300;
        opt.adaptive_lm = true; opt.nsub = 1;
        auto s = solver.solve(p.x0, xg, ug, opt);

        std::printf("fp-ddp %-14s status=%d iters=%3d cost=%.12e res_stat=%.2e res_eq=%.2e\n",
                    p.name.c_str(), s.status, s.iters, s.cost, s.statio, s.res_eq);

        std::string path = "logs/stress_" + p.name + "_fpddp.txt";
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) { std::printf("cannot open %s\n", path.c_str()); rc = 1; continue; }
        std::fprintf(f, "solver fp-ddp\nproblem %s\nN %d\ndt %.6f\nstatus %d\niters %d\ncost %.14e\nres_stat %.6e\nres_eq %.6e\n",
                     p.name.c_str(), p.N, p.dt, s.status, s.iters, s.cost, s.statio, s.res_eq);
        std::fprintf(f, "iterstats\n");           // it res_stat res_eq cost lm alpha qp_iter
        for (std::size_t i = 0; i < s.stats.size(); ++i) {
            const auto& st = s.stats[i];
            std::fprintf(f, "%zu %.10e %.10e %.14e %.10e %.6e %d\n",
                         i, st.res_stat, st.res_eq, st.cost, st.lm, st.alpha, st.qp_iter);
        }
        std::fprintf(f, "traj\n");
        for (int i = 0; i <= p.N; ++i) {
            std::fprintf(f, "%d", i);
            for (int k = 0; k < p.nx; ++k) std::fprintf(f, " %.10f", s.x[i][k]);
            for (int j = 0; j < p.nu; ++j) std::fprintf(f, " %.10f", i < p.N ? s.u[i][j] : 0.0);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        if (s.status != 0 && p.name != "cartpole_long") rc = 1;   // long horizon may hit GN limit
    }
    return rc;
}
