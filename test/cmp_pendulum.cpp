// ─────────────────────────────────────────────────────────────────────────────
//  test/cmp_pendulum.cpp — fp-ddp side of the acados comparison.
//  Shared problem (MUST match compare/acados_pendulum.py):
//    dynamics  ẋ1 = x2,  ẋ2 = −sin(x1) + u           (n=2, m=1, no control bounds)
//    horizon   T = 2.0, N = 40  (dt = 0.05), x0 = [2.0, 0.0]
//    cost      stage = ½(qx x1² + qv x2² + ru u²)·dt,  terminal = ½(qfx x1² + qfv x2²)
//              qx=1, qv=0.1, ru=0.01, qfx=qfv=50
//    method    DDP / Gauss-Newton, Gauss-Legendre IRK s=3, 1 step/interval.
//  Writes the solution to logs/cmp_fpddp.txt for compare.py.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdio>
#include <vector>

#include "fpddp/ddp.h"

using namespace fpddp;

struct PendDyn {
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = -std::sin(x[0]) + u[0]; }
    void fx(double, CSpan x, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=-std::cos(x[0]); A(1,1)=0; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct QuadCost {
    double dt, qx, qv, ru, qfx, qfv;
    double stage(double, CSpan x, CSpan u) const { return 0.5*(qx*x[0]*x[0] + qv*x[1]*x[1] + ru*u[0]*u[0])*dt; }
    void stage_grad(double, CSpan x, CSpan u, Span lx, Span lu) const { lx[0]=qx*x[0]*dt; lx[1]=qv*x[1]*dt; lu[0]=ru*u[0]*dt; }
    void stage_hess(double, CSpan, CSpan, Mat& Q, Mat& R, Mat&) const { Q(0,0)=qx*dt; Q(1,1)=qv*dt; R(0,0)=ru*dt; }
    double term(CSpan x) const { return 0.5*(qfx*x[0]*x[0] + qfv*x[1]*x[1]); }
    void term_grad(CSpan x, Span lx) const { lx[0]=qfx*x[0]; lx[1]=qfv*x[1]; }
    void term_hess(CSpan, Mat& Q) const { Q(0,0)=qfx; Q(1,1)=qfv; }
};

int main() {
    const int N = 40, nx = 2, nu = 1; const double dt = 0.05;
    std::vector<double> tg(N+1); for (int i=0;i<=N;++i) tg[i]=i*dt;
    Ddp solver(AnyDynamics(PendDyn{}), AnyCost(QuadCost{dt, 1.0, 0.1, 0.01, 50.0, 50.0}), N, nx, nu, tg);

    std::vector<std::vector<double>> u(N, std::vector<double>(nu, 0.0));
    std::vector<double> x0{2.0, 0.0};
    DdpOptions opt; opt.tol_stat = 1e-6; opt.max_iter = 300; opt.adaptive_lm = true;  // = acados script settings
    auto s = solver.solve(x0, u, opt);

    std::printf("fp-ddp: status=%d iters=%d cost=%.10e statio=%.2e xN=[%.6f %.6f]\n",
                s.status, s.iters, s.cost, s.statio, s.x[N][0], s.x[N][1]);

    FILE* f = std::fopen("logs/cmp_fpddp.txt", "w");
    if (f) {
        std::fprintf(f, "solver fp-ddp\nN %d\ndt %.6f\nstatus %d\niters %d\ncost %.12e\nstatio %.3e\ntraj\n",
                     N, dt, s.status, s.iters, s.cost, s.statio);
        for (int i = 0; i <= N; ++i) {
            double ui = (i < N) ? s.u[i][0] : 0.0;
            std::fprintf(f, "%d %.10f %.10f %.10f\n", i, s.x[i][0], s.x[i][1], ui);
        }
        std::fclose(f);
        std::printf("wrote logs/cmp_fpddp.txt\n");
    }
    return s.status == 0 ? 0 : 1;
}
