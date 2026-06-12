// ─────────────────────────────────────────────────────────────────────────────
//  test/globalization_test.cpp — exercise the ported acados globalization
//  policies (FixedStep, MeritBacktracking, Funnel) + adaptive LM on the
//  pendulum; the two backtracking globalizations must converge to the same
//  optimum.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdio>
#include <vector>

#include "fpddp/ddp.h"

using namespace fpddp;

struct PendDyn {
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = -std::sin(x[0]) - 0.1*x[1] + u[0]; }
    void fx(double, CSpan x, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=-std::cos(x[0]); A(1,1)=-0.1; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct QuadCost {
    double dt, qx, qv, ru, qfx, qfv;
    double stage(double, CSpan x, CSpan u) const { return 0.5*(qx*x[0]*x[0]+qv*x[1]*x[1]+ru*u[0]*u[0])*dt; }
    void stage_grad(double, CSpan x, CSpan u, Span lx, Span lu) const { lx[0]=qx*x[0]*dt; lx[1]=qv*x[1]*dt; lu[0]=ru*u[0]*dt; }
    void stage_hess(double, CSpan, CSpan, Mat& Q, Mat& R, Mat&) const { Q(0,0)=qx*dt; Q(1,1)=qv*dt; R(0,0)=ru*dt; }
    double term(CSpan x) const { return 0.5*(qfx*x[0]*x[0]+qfv*x[1]*x[1]); }
    void term_grad(CSpan x, Span lx) const { lx[0]=qfx*x[0]; lx[1]=qfv*x[1]; }
    void term_hess(CSpan, Mat& Q) const { Q(0,0)=qfx; Q(1,1)=qfv; }
};

int main() {
    std::printf("globalization policies + adaptive LM (pendulum)\n");
    const int N = 60, nx = 2, nu = 1; const double dt = 0.05;
    std::vector<double> tg(N+1); for (int i=0;i<=N;++i) tg[i]=i*dt;
    QuadCost cost{dt, 2,1, 0.05, 20,20};
    std::vector<std::vector<double>> u0(N, std::vector<double>(nu, 0.0));
    std::vector<double> x0{2.5, 0.0};

    int fail = 0;
    double merit_cost = 0, funnel_cost = 0;
    auto run = [&](const char* name, AnyGlobalization g) -> DdpSolution {
        Ddp solver(AnyDynamics(PendDyn{}), AnyCost(cost), N, nx, nu, tg);
        solver.globalization(std::move(g));
        DdpOptions opt; opt.tol_stat = 1e-6; opt.max_iter = 200; opt.adaptive_lm = true;
        auto s = solver.solve(x0, u0, opt);
        std::printf("  %-18s status=%d iters=%d cost=%.10e statio=%.1e\n", name, s.status, s.iters, s.cost, s.statio);
        return s;
    };

    auto sm = run("MeritBacktracking", MeritBacktracking{});  merit_cost  = sm.cost;
    auto sf = run("Funnel",            Funnel{});             funnel_cost = sf.cost;
    auto sx = run("FixedStep",         FixedStep{});

    auto check = [&](bool ok, const char* w){ std::printf("  [%s] %s\n", ok?"PASS":"FAIL", w); if(!ok) ++fail; };
    check(sm.status == 0 && sm.statio < 1e-5, "MeritBacktracking converges");
    check(sf.status == 0 && sf.statio < 1e-5, "Funnel converges");
    check(std::abs(merit_cost - funnel_cost) < 1e-7, "Merit and Funnel reach the same optimum");
    check(sx.status == 0 || sx.cost < merit_cost * 1.001, "FixedStep runs (full-step, may not converge tightly)");

    std::printf(fail==0 ? "\n globalization checks passed\n" : "\n %d FAILED\n", fail);
    return fail==0 ? 0 : 1;
}
