// smoke: AnyAny model interface + sim integrate_interval compile & run.
#include <cmath>
#include <cstdio>
#include <vector>
#include "fpddp/sim.h"

using namespace fpddp;

struct PendDyn {
    void rhs(double, CSpan x, CSpan u, Span dx) const { dx[0] = x[1]; dx[1] = -std::sin(x[0]) + u[0]; }
    void fx(double, CSpan x, CSpan, Mat& A) const { A(0,0)=0; A(0,1)=1; A(1,0)=-std::cos(x[0]); A(1,1)=0; }
    void fu(double, CSpan, CSpan, Mat& B) const { B(0,0)=0; B(1,0)=1; }
};

int main() {
    AnyDynamics dyn = PendDyn{};
    std::vector<double> x{0.5, 0.0}, u{0.1};
    auto r = integrate_interval(dyn, x, u, 0.0, 0.1, 1);
    std::printf("ok=%d  xnext=[%.6f %.6f]  A=[%.4f %.4f; %.4f %.4f]  B=[%.4f; %.4f]\n",
                int(r.ok), r.xnext[0], r.xnext[1],
                r.A(0,0), r.A(0,1), r.A(1,0), r.A(1,1), r.B(0,0), r.B(1,0));
    return r.ok ? 0 : 1;
}
