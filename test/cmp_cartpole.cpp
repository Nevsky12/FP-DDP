// ─────────────────────────────────────────────────────────────────────────────
//  test/cmp_cartpole.cpp — harder comparison: inverted-pendulum-on-a-cart
//  stabilization (acados' own pendulum_ode model: M=1, m=0.1, l=0.8, g=9.81;
//  θ=0 is upright/unstable).  4 states, nonlinear, unstable equilibrium.
//  Shared with compare/acados_cartpole.py.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdio>
#include <vector>

#include "fpddp/ddp.h"
#include "fpddp/fd_jacobian.h"

using namespace fpddp;

struct CartPole {
    static constexpr double M = 1.0, m = 0.1, g = 9.81, l = 0.8;
    void rhs(double, CSpan x, CSpan u, Span dx) const {
        const double th = x[1], om = x[3], F = u[0];
        const double ct = std::cos(th), st = std::sin(th);
        const double den = M + m - m * ct * ct;
        dx[0] = x[2];
        dx[1] = om;
        dx[2] = (-m * l * st * om * om + m * g * ct * st + F) / den;
        dx[3] = (-m * l * ct * st * om * om + F * ct + (M + m) * g * st) / (l * den);
    }
    void fx(double t, CSpan x, CSpan u, Mat& A) const {
        fd_state_jac([this](double tt, CSpan xx, CSpan uu, Span dd) { rhs(tt, xx, uu, dd); }, t, x, u, A);
    }
    void fu(double t, CSpan x, CSpan u, Mat& B) const {
        fd_control_jac([this](double tt, CSpan xx, CSpan uu, Span dd) { rhs(tt, xx, uu, dd); }, t, x, u, B);
    }
};

struct Quad4 {  // stage ½(Σ q_i x_i² + r u²)·dt ;  terminal ½ Σ qf_i x_i²
    double dt; double q[4]; double r; double qf[4];
    double stage(double, CSpan x, CSpan u) const {
        double c = r * u[0] * u[0];
        for (int i = 0; i < 4; ++i) c += q[i] * x[i] * x[i];
        return 0.5 * c * dt;
    }
    void stage_grad(double, CSpan x, CSpan u, Span lx, Span lu) const {
        for (int i = 0; i < 4; ++i) lx[i] = q[i] * x[i] * dt;
        lu[0] = r * u[0] * dt;
    }
    void stage_hess(double, CSpan, CSpan, Mat& Q, Mat& R, Mat&) const {
        for (int i = 0; i < 4; ++i) Q(i, i) = q[i] * dt;
        R(0, 0) = r * dt;
    }
    double term(CSpan x) const { double c = 0; for (int i = 0; i < 4; ++i) c += qf[i] * x[i] * x[i]; return 0.5 * c; }
    void term_grad(CSpan x, Span lx) const { for (int i = 0; i < 4; ++i) lx[i] = qf[i] * x[i]; }
    void term_hess(CSpan, Mat& Q) const { for (int i = 0; i < 4; ++i) Q(i, i) = qf[i]; }
};

int main() {
    const int N = 30, nx = 4, nu = 1; const double dt = 0.04;   // T = 1.2
    std::vector<double> tg(N + 1); for (int i = 0; i <= N; ++i) tg[i] = i * dt;
    Quad4 cost{dt, {10, 10, 0.1, 0.1}, 0.01, {100, 100, 10, 10}};
    Ddp solver(AnyDynamics(CartPole{}), AnyCost(cost), N, nx, nu, tg);

    std::vector<std::vector<double>> u(N, std::vector<double>(nu, 0.0));
    std::vector<double> x0{0.25, 0.08, 0.0, 0.0};   // mild offset; pole 0.08 rad off upright
    DdpOptions opt; opt.tol_stat = 1e-6; opt.max_iter = 300; opt.adaptive_lm = true;  // = acados script settings
    auto s = solver.solve(x0, u, opt);

    std::printf("fp-ddp cartpole: status=%d iters=%d cost=%.10e statio=%.2e xN=[%.4f %.4f %.4f %.4f]\n",
                s.status, s.iters, s.cost, s.statio, s.x[N][0], s.x[N][1], s.x[N][2], s.x[N][3]);

    FILE* f = std::fopen("logs/cmp_cartpole_fpddp.txt", "w");
    if (f) {
        std::fprintf(f, "solver fp-ddp\nproblem cartpole\nN %d\ndt %.6f\nstatus %d\niters %d\ncost %.12e\nstatio %.3e\ntraj\n",
                     N, dt, s.status, s.iters, s.cost, s.statio);
        for (int i = 0; i <= N; ++i) {
            double ui = (i < N) ? s.u[i][0] : 0.0;
            std::fprintf(f, "%d %.10f %.10f %.10f %.10f %.10f\n", i, s.x[i][0], s.x[i][1], s.x[i][2], s.x[i][3], ui);
        }
        std::fclose(f);
    }
    return s.status == 0 ? 0 : 1;
}
