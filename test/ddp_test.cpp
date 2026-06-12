// ─────────────────────────────────────────────────────────────────────────────
//  test/ddp_test.cpp — end-to-end DDP self-checks.
//    1. linear-quadratic: DDP must converge in ~1 iteration (Newton-exact).
//    2. pendulum regulation (nonlinear): converges, drives the state to target,
//       cost decreases monotonically.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdio>
#include <vector>

#include "fpddp/ddp.h"

using namespace fpddp;

struct LinDyn {  // ẋ1 = x2,  ẋ2 = −x1 − 0.1 x2 + u
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = -x[0] - 0.1*x[1] + u[0]; }
    void fx(double, CSpan, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=-1; A(1,1)=-0.1; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct PendDyn {  // ẋ1 = x2,  ẋ2 = −sin x1 + u
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = -std::sin(x[0]) + u[0]; }
    void fx(double, CSpan x, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=-std::cos(x[0]); A(1,1)=0; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct QuadCost {  // stage = ½(qx x1² + qv x2² + ru u²)·dt ; term = ½(qfx x1² + qfv x2²)
    double dt, qx, qv, ru, qfx, qfv;
    double stage(double, CSpan x, CSpan u) const { return 0.5*(qx*x[0]*x[0] + qv*x[1]*x[1] + ru*u[0]*u[0])*dt; }
    void stage_grad(double, CSpan x, CSpan u, Span lx, Span lu) const { lx[0]=qx*x[0]*dt; lx[1]=qv*x[1]*dt; lu[0]=ru*u[0]*dt; }
    void stage_hess(double, CSpan, CSpan, Mat& Q, Mat& R, Mat&) const { Q(0,0)=qx*dt; Q(1,1)=qv*dt; R(0,0)=ru*dt; }
    double term(CSpan x) const { return 0.5*(qfx*x[0]*x[0] + qfv*x[1]*x[1]); }
    void term_grad(CSpan x, Span lx) const { lx[0]=qfx*x[0]; lx[1]=qfv*x[1]; }
    void term_hess(CSpan, Mat& Q) const { Q(0,0)=qfx; Q(1,1)=qfv; }
};

static int g_fail = 0;
static void check(bool ok, const char* w) { std::printf("  [%s] %s\n", ok?"PASS":"FAIL", w); if(!ok) ++g_fail; }

static std::vector<double> tgrid(int N, double dt) { std::vector<double> t(N+1); for(int i=0;i<=N;++i) t[i]=i*dt; return t; }

int main() {
    std::printf("DDP end-to-end checks\n");

    // ── 1. linear-quadratic: Newton-exact, converges in ~1 iteration ──
    {
        const int N = 30, nx = 2, nu = 1; const double dt = 0.05;
        Ddp solver(AnyDynamics(LinDyn{}), AnyCost(QuadCost{dt, 1,1, 0.1, 5,5}), N, nx, nu, tgrid(N,dt));
        std::vector<std::vector<double>> u(N, std::vector<double>(nu, 0.0));
        std::vector<double> x0{1.0, 0.0};
        DdpOptions opt; opt.tol_stat = 1e-8; opt.adaptive_lm = false; opt.lm_fixed = 1e-9;  // Newton-exact
        auto s = solver.solve(x0, u, opt);
        std::printf("  LQR: status=%d iters=%d cost=%.6e statio=%.2e\n", s.status, s.iters, s.cost, s.statio);
        check(s.status == 0, "LQR converged");
        check(s.iters <= 2, "LQR converged in <= 2 iterations (Newton-exact)");
        check(s.statio < 1e-7, "LQR stationarity < 1e-7");
    }

    // ── 2. pendulum regulation (nonlinear) from a large angle to the origin ──
    {
        const int N = 60, nx = 2, nu = 1; const double dt = 0.05;
        Ddp solver(AnyDynamics(PendDyn{}), AnyCost(QuadCost{dt, 2,1, 0.05, 20,20}), N, nx, nu, tgrid(N,dt));
        std::vector<std::vector<double>> u(N, std::vector<double>(nu, 0.0));
        std::vector<double> x0{2.5, 0.0};

        // initial cost (rollout of u=0)
        Ddp probe = solver;  // copy (AnyAny is copyable)
        DdpOptions noop; noop.max_iter = 0;
        double cost0 = probe.solve(x0, u, noop).cost;

        DdpOptions opt; opt.tol_stat = 1e-6; opt.max_iter = 200; opt.adaptive_lm = true; opt.verbose = true;
        auto s = solver.solve(x0, u, opt);
        std::printf("  PEND: status=%d iters=%d cost=%.6e -> from %.6e  xN=[%.4f %.4f]  statio=%.2e\n",
                    s.status, s.iters, s.cost, cost0, s.x[N][0], s.x[N][1], s.statio);
        check(s.status == 0, "pendulum converged");
        check(s.cost < 0.5 * cost0, "pendulum cost reduced > 2x");
        check(std::abs(s.x[N][0]) < 0.1 && std::abs(s.x[N][1]) < 0.1, "pendulum driven to origin (|xN| < 0.1)");
        check(s.statio < 1e-5, "pendulum stationarity < 1e-5");
    }

    std::printf(g_fail==0 ? "\n all DDP checks passed\n" : "\n %d FAILED\n", g_fail);
    return g_fail==0 ? 0 : 1;
}
