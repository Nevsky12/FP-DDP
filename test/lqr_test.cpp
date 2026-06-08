// ─────────────────────────────────────────────────────────────────────────────
//  test/lqr_test.cpp — validate the hpipm OcpQp wrapper on a discrete LQR by
//  comparing the solved trajectory and the Riccati feedback K_0 against a
//  hand-coded backward Riccati recursion.  (nx=2, nu=1 keeps the hand algebra
//  scalar in the control.)
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdio>
#include <vector>

#include "irk/essence/algebra.hpp"
#include "irk/sensitivity.hpp"        // matmul
#include "fpddp/qp/ocp_qp.h"

using M = irk::essence::Matrix<double>;
using irk::sensitivity::matmul;

static M transp(const M& X) {
    M T(X.cols(), X.rows());
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) T(j, i) = X(i, j);
    return T;
}
static M sub(const M& A, const M& B) {
    M O(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) O(i, j) = A(i, j) - B(i, j);
    return O;
}
static M addm(const M& A, const M& B) {
    M O(A.rows(), A.cols());
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) O(i, j) = A(i, j) + B(i, j);
    return O;
}
// column-major flatten for hpipm
static std::vector<double> cm(const M& X) {
    std::vector<double> v;
    for (std::size_t j = 0; j < X.cols(); ++j)
        for (std::size_t i = 0; i < X.rows(); ++i) v.push_back(X(i, j));
    return v;
}

int main() {
    std::printf("LQR check: hpipm OcpQp vs hand Riccati\n");
    const int N = 10, nx = 2, nu = 1;

    M A(2, 2); A(0,0)=1; A(0,1)=0.1; A(1,0)=0;   A(1,1)=1;
    M B(2, 1); B(0,0)=0; B(1,0)=0.1;
    M Q(2, 2); Q(0,0)=1; Q(1,1)=1;
    M R(1, 1); R(0,0)=0.5;
    M Qf(2,2); Qf(0,0)=5; Qf(1,1)=5;
    std::vector<double> x0{1.0, 0.0};

    // ── hand backward Riccati: P_N = Qf; K_i, P_i ──
    std::vector<M> Kg(N, M(1, 2));
    M P = Qf;
    for (int i = N - 1; i >= 0; --i) {
        M Bt = transp(B), At = transp(A);
        M BtP = matmul(Bt, P);              // 1×2
        M S1  = addm(R, matmul(BtP, B));    // 1×1
        double sinv = 1.0 / S1(0,0);
        M BtPA = matmul(BtP, A);            // 1×2
        M K(1,2); K(0,0)=sinv*BtPA(0,0); K(0,1)=sinv*BtPA(0,1);   // K = S^{-1} B'PA
        Kg[i] = K;
        M AtP = matmul(At, P);              // 2×2
        M AtPA = matmul(AtP, A);            // 2×2
        M AtPB = matmul(AtP, B);            // 2×1
        P = addm(Q, sub(AtPA, matmul(AtPB, K)));  // P = Q + A'PA - A'PB K
    }
    // hand forward rollout: u_i = -K_i x_i
    std::vector<std::vector<double>> xh(N+1), uh(N);
    xh[0] = x0;
    for (int i = 0; i < N; ++i) {
        double u = -(Kg[i](0,0)*xh[i][0] + Kg[i](0,1)*xh[i][1]);
        uh[i] = {u};
        xh[i+1] = { A(0,0)*xh[i][0]+A(0,1)*xh[i][1] + B(0,0)*u,
                    A(1,0)*xh[i][0]+A(1,1)*xh[i][1] + B(1,0)*u };
    }

    // ── hpipm solve ──
    fpddp::OcpQp qp(N, std::vector<int>(N+1, nx), [&]{ auto v=std::vector<int>(N+1,nu); v[N]=0; return v; }());
    auto cmA=cm(A), cmB=cm(B), cmQ=cm(Q), cmR=cm(R), cmQf=cm(Qf);
    std::vector<double> zb(nx,0.0), zq(nx,0.0), zr(nu,0.0), zS(nu*nx,0.0);
    for (int i = 0; i < N; ++i) {
        qp.setA(i, cmA.data()); qp.setB(i, cmB.data()); qp.setb(i, zb.data());
        qp.setQ(i, cmQ.data()); qp.setR(i, cmR.data()); qp.setS(i, zS.data());
        qp.setq(i, zq.data());  qp.setr(i, zr.data());
    }
    qp.setQ(N, cmQf.data()); qp.setq(N, zq.data());
    qp.setx0(x0.data());

    int st = qp.solve();
    std::printf("  hpipm status = %d, iter = %d\n", st, qp.iter());
    if (st != 0) { std::printf("  [FAIL] hpipm did not solve (status %d)\n", st); return 1; }

    // compare trajectories
    double xerr = 0, uerr = 0;
    std::vector<double> xb(nx), ub(nu);
    for (int i = 0; i <= N; ++i) {
        qp.getx(i, xb.data());
        for (int k = 0; k < nx; ++k) xerr = std::max(xerr, std::abs(xb[k] - xh[i][k]));
    }
    for (int i = 0; i < N; ++i) {
        qp.getu(i, ub.data());
        uerr = std::max(uerr, std::abs(ub[0] - uh[i][0]));
    }

    // compare feedback K_0 (hpipm sign may be ±; the LQR optimum is u=-Kg x)
    std::vector<double> K0(nu*nx);
    qp.getK(0, K0.data());            // 1×2 col-major → [K(0,0), K(0,1)]
    double kp = std::max(std::abs(K0[0]-Kg[0](0,0)), std::abs(K0[1]-Kg[0](0,1)));
    double km = std::max(std::abs(K0[0]+Kg[0](0,0)), std::abs(K0[1]+Kg[0](0,1)));
    double kerr = std::min(kp, km);

    std::printf("  hpipm K0 = [%.6f %.6f], hand Kgain0 = [%.6f %.6f] (sign %s)\n",
                K0[0], K0[1], Kg[0](0,0), Kg[0](0,1), kp < km ? "+" : "-");

    int fail = 0;
    auto check = [&](bool ok, const char* w, double e){ std::printf("  [%s] %s (%.2e)\n", ok?"PASS":"FAIL", w, e); if(!ok) ++fail; };
    check(xerr < 1e-7, "state trajectory x matches hand Riccati", xerr);
    check(uerr < 1e-7, "control trajectory u matches hand Riccati", uerr);
    check(kerr < 1e-7, "Riccati feedback K0 matches (up to sign)", kerr);

    std::printf(fail==0 ? "\n LQR check passed\n" : "\n %d FAILED\n", fail);
    return fail==0?0:1;
}
