// ─────────────────────────────────────────────────────────────────────────────
//  test/exact_hessian_test.cpp — exact-Hessian DDP self-checks.
//    1. LQ: dynamics curvature is zero ⇒ exact ≡ Gauss-Newton (same optimum,
//       Newton-exact iteration count unchanged).
//    2. hard pendulum (x0 = [3.1, 0], weak control penalty): exact Hessian
//       reaches the same optimum in no more iterations than GN, and converges
//       to a tight tolerance (quadratic tail) where GN grinds.
//    3. stiff Van der Pol (μ = 3): same optimum, iteration count strictly
//       reduced vs GN.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdio>
#include <vector>

#include "fpddp/ddp.h"

using namespace fpddp;

struct LinDyn {
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = -x[0] - 0.1*x[1] + u[0]; }
    void fx(double, CSpan, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=-1; A(1,1)=-0.1; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct PendDyn {
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = -std::sin(x[0]) + u[0]; }
    void fx(double, CSpan x, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=-std::cos(x[0]); A(1,1)=0; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};
struct VdpDyn {
    double mu = 3.0;
    void rhs(double, CSpan x, CSpan u, Span dx) const {
        dx[0] = x[1]; dx[1] = mu*(1.0 - x[0]*x[0])*x[1] - x[0] + u[0];
    }
    void fx(double, CSpan x, CSpan, Mat& A) const {
        A(0,0)=0; A(0,1)=1; A(1,0)=-2.0*mu*x[0]*x[1]-1.0; A(1,1)=mu*(1.0-x[0]*x[0]);
    }
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

static int g_fail = 0;
static void check(bool ok, const char* w) { std::printf("  [%s] %s\n", ok?"PASS":"FAIL", w); if(!ok) ++g_fail; }
static std::vector<double> tgrid(int N, double dt) { std::vector<double> t(N+1); for(int i=0;i<=N;++i) t[i]=i*dt; return t; }

template <class Dyn>
static std::pair<DdpSolution, DdpSolution> run_both(Dyn dyn, QuadCost cost, int N, double dt,
                                                    std::vector<double> x0, double tol,
                                                    double switch_stat) {
    std::vector<std::vector<double>> xg(N+1, x0), ug(N, std::vector<double>(1, 0.0));
    DdpOptions gn;  gn.tol_stat = tol; gn.max_iter = 300;          // LM off: pure GN vs pure Newton
    DdpOptions ex = gn; ex.hessian = HessianMode::exact_fd; ex.exact_switch_stat = switch_stat;
    Ddp s1(AnyDynamics(dyn), AnyCost(cost), N, 2, 1, tgrid(N, dt));
    Ddp s2(AnyDynamics(dyn), AnyCost(cost), N, 2, 1, tgrid(N, dt));
    auto a = s1.solve(x0, xg, ug, gn);
    auto b = s2.solve(x0, xg, ug, ex);
    return {a, b};
}

int main() {
    std::printf("exact-Hessian DDP checks\n");

    {   // 1. LQ: curvature-free ⇒ identical behavior
        auto [gn, ex] = run_both(LinDyn{}, QuadCost{0.05, 1,1, 0.1, 5,5}, 30, 0.05, {1.0, 0.0}, 1e-8, 1e300);
        std::printf("  LQ        : GN %d its cost %.10e | exact %d its cost %.10e\n",
                    gn.iters, gn.cost, ex.iters, ex.cost);
        check(gn.status == 0 && ex.status == 0, "LQ both converge");
        check(std::abs(gn.cost - ex.cost) < 1e-10, "LQ same optimum");
        check(ex.iters == gn.iters, "LQ iteration count unchanged");
    }
    {   // 2. hard pendulum, tight tolerance: exact must not lose to GN
        auto [gn, ex] = run_both(PendDyn{}, QuadCost{0.05, 1,0.1, 0.001, 100,100}, 60, 0.05, {3.1, 0.0}, 1e-9, 1e300);
        std::printf("  pend_hard : GN %d its cost %.10e statio %.1e | exact %d its cost %.10e statio %.1e\n",
                    gn.iters, gn.cost, gn.statio, ex.iters, ex.cost, ex.statio);
        check(ex.status == 0, "pendulum exact converges to 1e-9");
        check(gn.status != 0 || std::abs(gn.cost - ex.cost) / std::max(1.0, std::abs(ex.cost)) < 1e-7,
              "pendulum same optimum (when GN converges)");
        check(gn.status != 0 || ex.iters <= gn.iters, "pendulum exact needs <= GN iterations");
        check(gn.status != 0 || ex.statio < gn.statio, "pendulum exact reaches tighter stationarity (quadratic tail)");
    }
    {   // 3. stiff VdP, hybrid GN→exact switch: the exact stage Hessian is
        //     indefinite (handed to hpipm raw; projected only if the QP fails) —
        //     quadratic tail must beat GN decisively.
        auto [gn, ex] = run_both(VdpDyn{3.0}, QuadCost{0.10, 1,1, 0.01, 10,10}, 60, 0.10, {3.0, 1.0}, 1e-7, 1e-1);
        std::printf("  vdp_hard  : GN %d its cost %.10e statio %.1e | exact %d its cost %.10e statio %.1e\n",
                    gn.iters, gn.cost, gn.statio, ex.iters, ex.cost, ex.statio);
        check(gn.status == 0 && ex.status == 0, "vdp both converge");
        check(std::abs(gn.cost - ex.cost) / std::max(1.0, std::abs(ex.cost)) < 1e-7, "vdp same optimum");
        check(ex.iters < gn.iters, "vdp exact strictly fewer iterations (quadratic tail)");
    }

    std::printf(g_fail==0 ? "\n exact-Hessian checks passed\n" : "\n %d FAILED\n", g_fail);
    return g_fail==0 ? 0 : 1;
}
