// ─────────────────────────────────────────────────────────────────────────────
//  test/cmp_unicycle.cpp — multi-input comparison: a nonholonomic unicycle
//  (3 states, 2 controls) regulated from an offset pose to the origin.
//    ẋ = v·cosθ,  ẏ = v·sinθ,  θ̇ = ω      x=[px,py,θ], u=[v,ω]
//  Shared with compare/acados_unicycle.py.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdio>
#include <vector>

#include "fpddp/ddp.h"
#include "fpddp/fd_jacobian.h"

using namespace fpddp;

struct Unicycle {
    void rhs(double, CSpan x, CSpan u, Span dx) const {
        dx[0] = u[0] * std::cos(x[2]);
        dx[1] = u[0] * std::sin(x[2]);
        dx[2] = u[1];
    }
    void fx(double t, CSpan x, CSpan u, Mat& A) const {
        fd_state_jac([this](double tt, CSpan xx, CSpan uu, Span dd) { rhs(tt, xx, uu, dd); }, t, x, u, A);
    }
    void fu(double t, CSpan x, CSpan u, Mat& B) const {
        fd_control_jac([this](double tt, CSpan xx, CSpan uu, Span dd) { rhs(tt, xx, uu, dd); }, t, x, u, B);
    }
};

struct Quad3x2 {  // stage ½(Σ q_i x_i² + Σ r_j u_j²)·dt ; terminal ½ Σ qf_i x_i²
    double dt; double q[3]; double r[2]; double qf[3];
    double stage(double, CSpan x, CSpan u) const {
        double c = 0; for (int i=0;i<3;++i) c += q[i]*x[i]*x[i]; for (int j=0;j<2;++j) c += r[j]*u[j]*u[j];
        return 0.5*c*dt;
    }
    void stage_grad(double, CSpan x, CSpan u, Span lx, Span lu) const {
        for (int i=0;i<3;++i) lx[i]=q[i]*x[i]*dt; for (int j=0;j<2;++j) lu[j]=r[j]*u[j]*dt;
    }
    void stage_hess(double, CSpan, CSpan, Mat& Q, Mat& R, Mat&) const {
        for (int i=0;i<3;++i) Q(i,i)=q[i]*dt; for (int j=0;j<2;++j) R(j,j)=r[j]*dt;
    }
    double term(CSpan x) const { double c=0; for (int i=0;i<3;++i) c+=qf[i]*x[i]*x[i]; return 0.5*c; }
    void term_grad(CSpan x, Span lx) const { for (int i=0;i<3;++i) lx[i]=qf[i]*x[i]; }
    void term_hess(CSpan, Mat& Q) const { for (int i=0;i<3;++i) Q(i,i)=qf[i]; }
};

int main() {
    const int N = 40, nx = 3, nu = 2; const double dt = 0.05;   // T = 2.0
    std::vector<double> tg(N+1); for (int i=0;i<=N;++i) tg[i]=i*dt;
    Quad3x2 cost{dt, {1,1,0.5}, {0.1,0.1}, {50,50,20}};
    Ddp solver(AnyDynamics(Unicycle{}), AnyCost(cost), N, nx, nu, tg);

    std::vector<std::vector<double>> u(N, std::vector<double>(nu, 0.0));
    std::vector<double> x0{1.0, 0.5, 0.8};   // offset pose -> regulate to origin
    DdpOptions opt; opt.tol = 1e-7; opt.max_iter = 300;
    auto s = solver.solve(x0, u, opt);

    std::printf("fp-ddp unicycle: status=%d iters=%d cost=%.10e statio=%.2e xN=[%.4f %.4f %.4f]\n",
                s.status, s.iters, s.cost, s.statio, s.x[N][0], s.x[N][1], s.x[N][2]);

    FILE* f = std::fopen("logs/cmp_unicycle_fpddp.txt", "w");
    if (f) {
        std::fprintf(f, "solver fp-ddp\nproblem unicycle\nN %d\ndt %.6f\nstatus %d\niters %d\ncost %.12e\nstatio %.3e\ntraj\n",
                     N, dt, s.status, s.iters, s.cost, s.statio);
        for (int i = 0; i <= N; ++i) {
            double u0 = (i<N)? s.u[i][0]:0.0, u1 = (i<N)? s.u[i][1]:0.0;
            std::fprintf(f, "%d %.10f %.10f %.10f %.10f %.10f\n", i, s.x[i][0], s.x[i][1], s.x[i][2], u0, u1);
        }
        std::fclose(f);
    }
    return s.status == 0 ? 0 : 1;
}
