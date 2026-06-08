#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · ddp.h
//
//  Feasible-path DDP / iLQR, structured to mirror acados' ocp_nlp_ddp steps:
//    per iteration —
//      1. adaptive Levenberg-Marquardt:  lm = obj_scalar·cost·μ, with μ decreased
//         on accepted full steps (÷λ, one-step memory) and increased on backtracks
//         (×λ, capped 1), floor μ_min  (acados adaptive_levenberg_marquardt).
//      2. linearize every shooting interval (sim → A,B, defect b) + cost grad/Hess,
//         assemble + LM-regularize the hpipm stage QP.
//      3. stationarity: backward costate (adjoint) sweep → reduced gradient g_i and
//         the discrete costates λ_i (dual seed).
//      4. backward pass: hpipm Riccati QP → feedback K_i, feedforward k_i.
//      5. predicted reduction from the (first-order) QP model.
//      6. forward pass: feasible nonlinear rollout u=ū+αk+K δx with an
//         AnyGlobalization line search (FixedStep | MeritBacktracking | Funnel) —
//         Armijo sufficient-descent (actual ≥ ε·α·predicted).
//  Only the initial state is constrained (faithful to acados DDP).
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

#include "fpddp/model.h"
#include "fpddp/sim.h"
#include "fpddp/qp/ocp_qp.h"
#include "fpddp/globalization.h"

namespace fpddp {

struct DdpOptions {
    int    max_iter = 200;
    double tol      = 1e-7;        // max-norm of the reduced gradient g_i
    // adaptive Levenberg-Marquardt (acados):  lm = lm_obj_scalar · cost · μ
    bool   adaptive_lm   = true;
    double lm_mu0        = 1e-3;
    double lm_mu_min     = 1e-16;
    double lm_lam        = 5.0;
    double lm_obj_scalar = 2.0;
    double lm_fixed      = 1e-6;   // used when adaptive_lm == false
    // line search
    double alpha_min       = 1e-8;
    double alpha_reduction = 0.5;
    int    ls_max          = 40;
    std::size_t nsub       = 1;    // IRK substeps per shooting interval
    bool   verbose         = false;
};

struct DdpSolution {
    std::vector<std::vector<double>> x, u;       // x: N+1 × nx,  u: N × nu
    std::vector<std::vector<double>> costate;    // λ: N+1 × nx  (dual seed)
    double cost   = 0;
    double statio = 0;
    int    iters  = 0;
    int    status = 0;             // 0 converged, 1 maxiter, 2 qp_fail, 3 line-search fail
};

class Ddp {
public:
    Ddp(AnyDynamics dyn, AnyCost cost, int N, int nx, int nu, std::vector<double> tgrid)
        : dyn_(std::move(dyn)), cost_(std::move(cost)),
          N_(N), nx_(nx), nu_(nu), t_(std::move(tgrid)), glob_(MeritBacktracking{}) {}

    void globalization(AnyGlobalization g) { glob_ = std::move(g); }

    DdpSolution solve(std::span<const double> x0, std::vector<std::vector<double>> u) {
        return solve(x0, std::move(u), DdpOptions{});
    }

    DdpSolution solve(std::span<const double> x0, std::vector<std::vector<double>> u, DdpOptions opt) {
        DdpSolution sol;
        std::vector<std::vector<double>> x(N_ + 1);
        x[0].assign(x0.begin(), x0.end());
        for (int i = 0; i < N_; ++i)
            x[i + 1] = integrate_interval(dyn_, x[i], u[i], t_[i], t_[i + 1], opt.nsub).xnext;
        double cost = total_cost(x, u);

        OcpQp qp(N_, std::vector<int>(N_ + 1, nx_), nu_vec());
        std::vector<double> zx0(nx_, 0.0);
        glob_.restart(0.0);                                 // feasible rollout → infeasibility 0

        double mu = opt.lm_mu0, mu_bar = opt.lm_mu0, prev_alpha = 1.0, statio = 0;
        std::vector<Mat> A(N_), B(N_);
        std::vector<std::vector<double>> q(N_ + 1), r(N_), lam(N_ + 1), Kc(N_), kc(N_);

        int it = 0;
        for (; it < opt.max_iter; ++it) {
            // 1. adaptive Levenberg-Marquardt
            double lm;
            if (opt.adaptive_lm) {
                if (it == 0)                 { mu = mu_bar = opt.lm_mu0; }
                else if (prev_alpha == 1.0)  { double t = mu; mu = std::max(opt.lm_mu_min, mu_bar / opt.lm_lam); mu_bar = t; }
                else                         { mu = std::min(opt.lm_lam * mu, 1.0); }
                lm = opt.lm_obj_scalar * cost * mu;
            } else {
                lm = opt.lm_fixed;
            }

            // 2. linearize + assemble QP
            for (int i = 0; i < N_; ++i) {
                auto s = integrate_interval(dyn_, x[i], u[i], t_[i], t_[i + 1], opt.nsub);
                A[i] = s.A; B[i] = s.B;
                std::vector<double> b(nx_);
                for (int k = 0; k < nx_; ++k) b[k] = s.xnext[k] - x[i + 1][k];
                Mat Q(nx_, nx_), R(nu_, nu_), S(nu_, nx_);
                q[i].assign(nx_, 0.0); r[i].assign(nu_, 0.0);
                cost_.stage_hess(t_[i], x[i], u[i], Q, R, S);
                cost_.stage_grad(t_[i], x[i], u[i], q[i], r[i]);
                Mat Qr = Q, Rr = R;
                for (int k = 0; k < nx_; ++k) Qr(k, k) += lm;
                for (int k = 0; k < nu_; ++k) Rr(k, k) += lm;
                auto cA = colmajor(A[i]), cB = colmajor(B[i]), cQ = colmajor(Qr), cR = colmajor(Rr), cS = colmajor(S);
                qp.setA(i, cA.data()); qp.setB(i, cB.data()); qp.setb(i, b.data());
                qp.setQ(i, cQ.data()); qp.setR(i, cR.data()); qp.setS(i, cS.data());
                qp.setq(i, q[i].data()); qp.setr(i, r[i].data());
            }
            Mat QN(nx_, nx_);
            q[N_].assign(nx_, 0.0);
            cost_.term_hess(x[N_], QN);
            cost_.term_grad(x[N_], q[N_]);
            { Mat QNr = QN; for (int k = 0; k < nx_; ++k) QNr(k, k) += lm;
              auto cQN = colmajor(QNr); qp.setQ(N_, cQN.data()); qp.setq(N_, q[N_].data()); }

            // 3. stationarity via backward costate sweep
            lam[N_] = q[N_];
            statio = 0;
            for (int i = N_ - 1; i >= 0; --i) {
                std::vector<double> g = r[i];
                for (int a = 0; a < nu_; ++a) { double s = 0; for (int p = 0; p < nx_; ++p) s += B[i](p, a) * lam[i + 1][p]; g[a] += s; }
                for (int a = 0; a < nu_; ++a) statio = std::max(statio, std::abs(g[a]));
                lam[i] = q[i];
                for (int b = 0; b < nx_; ++b) { double s = 0; for (int p = 0; p < nx_; ++p) s += A[i](p, b) * lam[i + 1][p]; lam[i][b] += s; }
            }
            if (opt.verbose)
                std::printf("  it %3d  cost %.8e  statio %.3e  lm %.2e  mu %.2e\n", it, cost, statio, lm, mu);
            if (statio < opt.tol) { sol.status = 0; break; }

            // 4. backward pass (hpipm)
            qp.setx0(zx0.data());
            if (qp.solve() != 0) { sol.status = 2; break; }
            for (int i = 0; i < N_; ++i) {
                Kc[i].assign(nu_ * nx_, 0.0); kc[i].assign(nu_, 0.0);
                qp.getK(i, Kc[i].data()); qp.getk(i, kc[i].data());
            }

            // 5. predicted reduction (first-order: −gᵀδ over the linear QP feedback rollout, b≈0)
            double gd = 0.0;
            { std::vector<double> dx(nx_, 0.0);
              for (int i = 0; i < N_; ++i) {
                  std::vector<double> du(nu_);
                  for (int a = 0; a < nu_; ++a) { double s = kc[i][a]; for (int b = 0; b < nx_; ++b) s += Kc[i][a + b * nu_] * dx[b]; du[a] = s; }
                  for (int b = 0; b < nx_; ++b) gd += q[i][b] * dx[b];
                  for (int a = 0; a < nu_; ++a) gd += r[i][a] * du[a];
                  std::vector<double> nd(nx_, 0.0);
                  for (int p = 0; p < nx_; ++p) { double s = 0; for (int b = 0; b < nx_; ++b) s += A[i](p, b) * dx[b]; for (int a = 0; a < nu_; ++a) s += B[i](p, a) * du[a]; nd[p] = s; }
                  dx = nd;
              }
              for (int b = 0; b < nx_; ++b) gd += q[N_][b] * dx[b];
            }
            const double pred_red = -gd;

            // 6. forward pass: feasible rollout + AnyGlobalization line search
            bool accepted = false;
            std::vector<std::vector<double>> xn, un;
            double cn = 0, alpha = 1.0;
            for (int ls = 0; ls < opt.ls_max && alpha > opt.alpha_min; ++ls) {
                xn.assign(N_ + 1, {}); un.assign(N_, {});
                xn[0] = x[0];
                bool ok = true;
                for (int i = 0; i < N_; ++i) {
                    std::vector<double> du(nu_);
                    for (int a = 0; a < nu_; ++a) {
                        double s = alpha * kc[i][a];
                        for (int b = 0; b < nx_; ++b) s += Kc[i][a + b * nu_] * (xn[i][b] - x[i][b]);
                        du[a] = s;
                    }
                    un[i].resize(nu_);
                    for (int a = 0; a < nu_; ++a) un[i][a] = u[i][a] + du[a];
                    auto s = integrate_interval(dyn_, xn[i], un[i], t_[i], t_[i + 1], opt.nsub);
                    if (!s.ok) { ok = false; break; }
                    xn[i + 1] = s.xnext;
                }
                if (ok) {
                    cn = total_cost(xn, un);
                    if (glob_.accept(cost, cn, pred_red, alpha, 0.0, 0.0)) { accepted = true; break; }
                }
                alpha *= opt.alpha_reduction;
            }
            if (!accepted) { sol.status = 3; break; }
            x = std::move(xn); u = std::move(un); cost = cn;
            prev_alpha = alpha; glob_.notify(alpha, 0.0);
        }
        if (it >= opt.max_iter) sol.status = 1;

        sol.x = std::move(x); sol.u = std::move(u); sol.costate = std::move(lam);
        sol.cost = cost; sol.iters = it; sol.statio = statio;
        return sol;
    }

private:
    AnyDynamics dyn_; AnyCost cost_;
    int N_, nx_, nu_;
    std::vector<double> t_;
    AnyGlobalization glob_;

    std::vector<int> nu_vec() const { std::vector<int> v(N_ + 1, nu_); v[N_] = 0; return v; }

    double total_cost(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& u) {
        double c = 0;
        for (int i = 0; i < N_; ++i) c += cost_.stage(t_[i], x[i], u[i]);
        c += cost_.term(x[N_]);
        return c;
    }
};

}  // namespace fpddp
