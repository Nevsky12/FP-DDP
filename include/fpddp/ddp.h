#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · ddp.h
//
//  Feasible-path DDP mirroring acados' ocp_nlp_ddp (v0.5.4) step for step.
//  Per iteration:
//    1. linearize every shooting interval (sim → A,B, defect b) + cost grad/Hess.
//    2. adaptive Levenberg-Marquardt:  lm = obj_scalar·cost·μ, μ decreased on
//       full steps (÷λ, one-step memory μ̄), increased otherwise (×λ, cap 1),
//       floor μ_min  (acados adaptive_levenberg_marquardt_update_mu); the LM
//       term lm·scaling_i·I is added to the QP Hessian diagonals.
//    3. NLP residuals with the *carried duals* (full-step dual update, the
//       acados DDP default):  res_stat = ∇L per (u,x) block with π from the
//       last QP, res_eq = max‖f(xᵢ,uᵢ) − xᵢ₊₁‖∞  (acados ocp_nlp_res_compute).
//    4. termination in acados' check_termination order: NaN → converged
//       (eq && stat) → zero-residual LS → small step → max iterations.
//    5. hpipm Riccati QP → feedback Kᵢ, feedforward kᵢ, QP solution d=(δx,δu),
//       duals; step_norm = ‖d‖∞; predicted reductions pred_qp = −qpobj
//       (incl. ½dᵀHd, acados DDP merit) and pred_lin = −gᵀd (acados funnel).
//    6. forward pass: feasible rollout u = ū + αk + K(x − x̄), x⁺ = f(x,u) with
//       an AnyGlobalization backtracking loop (α ← α·red until α < α_min).
//       A dynamically infeasible initial guess takes one forced full step
//       (acados' infeasible_initial_guess handling).
//  Only the initial state is constrained (as acados DDP).
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

#include "fpddp/model.h"
#include "fpddp/sim.h"
#include "fpddp/irk_stepper.h"
#include "fpddp/regularize.h"
#include "fpddp/qp/ocp_qp.h"
#include "fpddp/globalization.h"

namespace fpddp {

// Hessian of the QP: Gauss-Newton (cost curvature only, the acados DDP mode) or
// exact — GN plus the dynamics curvature Σ_k π_k ∂²Φ_k/∂(x,u)², computed
// automatically by central differences of the exact adjoint gradients
// [Aᵀπ; Bᵀπ] (no user-supplied second derivatives).
enum class HessianMode { gauss_newton, exact_fd };

struct DdpOptions {
    int    max_iter = 100;          // acados nlp_solver_max_iter default
    double tol_stat = 1e-6;         // acados nlp_solver_tol_stat default
    double tol_eq   = 1e-6;         // acados nlp_solver_tol_eq default
    double tol_zero_res = 0.0;      // acados leaves tol_zero_res uninitialized (calloc ⇒ 0.0)
    // Levenberg-Marquardt (acados ocp_nlp_opts; adaptive off by default as in acados)
    bool   adaptive_lm   = false;   // with_adaptive_levenberg_marquardt
    double lm_mu0        = 1e-3;
    double lm_mu_min     = 1e-16;
    double lm_lam        = 5.0;
    double lm_obj_scalar = 2.0;
    double lm_fixed      = 0.0;     // levenberg_marquardt (used when !adaptive_lm)
    std::vector<double> lm_scaling; // per-stage cost scaling for the LM term (N+1 entries;
                                    // empty ⇒ all 1, the discrete-cost convention; acados
                                    // default LINEAR_LS scaling is Δt per stage)
    // line search (acados Python-interface defaults for DDP)
    double alpha_min       = 1e-17;
    double alpha_reduction = 0.7;
    std::size_t nsub       = 1;     // IRK substeps per shooting interval
    // exact-Hessian DDP
    HessianMode hessian    = HessianMode::gauss_newton;
    double fd_hess_eps     = 1e-4;  // central-difference step (scaled per component)
    double exact_switch_stat = 1e-1;// enable curvature once res_stat < this (∞: always, 0: never);
                                    // GN globalizes better far out, exact Newton wins the endgame
    double reg_min_eig     = 1e-8;  // eigenvalue floor of the projected stage Hessian
    bool   reg_mirror      = true;  // mirror (|λ|) vs project (ε-floor) regularization
    bool   verbose         = false;
};

// per-iteration statistics, acados `stat` analog.  Row i holds the residuals /
// cost of iterate i plus the LM, QP and step data of the iteration *taken from*
// iterate i (acados stores qp data on row i+1 and lm with one-row delay).
struct IterStat {
    double res_stat = 0, res_eq = 0, cost = 0, lm = 0;
    double alpha = 0, step_norm = 0;
    int    qp_status = -1, qp_iter = 0;
};

struct DdpSolution {
    std::vector<std::vector<double>> x, u;       // x: N+1 × nx,  u: N × nu
    std::vector<std::vector<double>> costate;    // λ: N+1 × nx, exact adjoint at the final iterate
    double cost     = 0;
    double statio   = 0;           // inf-norm stationarity residual (acados res_stat)
    double res_eq   = 0;           // inf-norm dynamics defect (acados res_eq)
    int    iters    = 0;
    int    status   = 0;           // 0 converged, 1 maxiter, 2 qp_fail,
                                   // 3 line-search fail (min step), 4 NaN detected,
                                   // 5 small step (acados ACADOS_MINSTEP)
    std::vector<IterStat> stats;
};

class Ddp {
public:
    Ddp(AnyDynamics dyn, AnyCost cost, int N, int nx, int nu, std::vector<double> tgrid)
        : dyn_(std::move(dyn)), cost_(std::move(cost)),
          N_(N), nx_(nx), nu_(nu), t_(std::move(tgrid)), glob_(MeritBacktracking{}) {}

    void globalization(AnyGlobalization g) { glob_ = std::move(g); }

    // feasible-path entry: roll the given controls out from x0 (a feasible guess).
    DdpSolution solve(std::span<const double> x0, std::vector<std::vector<double>> u) {
        return solve(x0, std::move(u), DdpOptions{});
    }
    DdpSolution solve(std::span<const double> x0, std::vector<std::vector<double>> u, DdpOptions opt) {
        std::vector<std::vector<double>> x(N_ + 1);
        x[0].assign(x0.begin(), x0.end());
        for (int i = 0; i < N_; ++i)
            x[i + 1] = integrate_interval(dyn_, x[i], u[i], t_[i], t_[i + 1], opt.nsub).xnext;
        return solve_impl(x0, std::move(x), std::move(u), opt);
    }
    // acados-style entry: arbitrary primal guess (x_guess need not satisfy the
    // dynamics or start at x0; acados' default initialization is xᵢ ≡ x0, u ≡ 0).
    DdpSolution solve(std::span<const double> x0, std::vector<std::vector<double>> x_guess,
                      std::vector<std::vector<double>> u_guess, DdpOptions opt) {
        return solve_impl(x0, std::move(x_guess), std::move(u_guess), opt);
    }

private:
    DdpSolution solve_impl(std::span<const double> x0, std::vector<std::vector<double>> x,
                           std::vector<std::vector<double>> u, DdpOptions opt) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        DdpSolution sol;
        double cost = total_cost(x, u);

        OcpQp qp(N_, std::vector<int>(N_ + 1, nx_), nu_vec());

        // linearization storage (kept for residuals, QP objective and costates)
        std::vector<Mat> A(N_), B(N_), Qr(N_), Rr(N_), S(N_);
        Mat QNr(nx_, nx_);
        std::vector<std::vector<double>> q(N_ + 1), r(N_), b(N_);
        // current duals (zero-initialized; replaced by QP duals after each step — full_step_dual)
        std::vector<std::vector<double>> pi(N_, std::vector<double>(nx_, 0.0));
        std::vector<double> lam_lb0(nx_, 0.0), lam_ub0(nx_, 0.0);
        // QP solution
        std::vector<std::vector<double>> dx(N_ + 1), du(N_), Kc(N_), kc(N_), pi_qp(N_);
        std::vector<double> lamlb_qp(nx_), lamub_qp(nx_), x0bound(nx_);

        double mu = opt.lm_mu0, mu_bar = opt.lm_mu0, prev_alpha = 0.0;
        double step_norm = 0.0, res_stat_prev = std::numeric_limits<double>::infinity();
        bool curvature_on = false;
        bool infeasible_guess = true;     // resolved from res_eq at iteration 0
        const auto lm_scale = [&](int i) {
            return opt.lm_scaling.empty() ? 1.0 : opt.lm_scaling[std::size_t(i)];
        };

        // per-interval steppers: warm-started across iterations and line-search trials
        std::vector<IrkStepper> steppers;
        steppers.reserve(std::size_t(N_));
        for (int i = 0; i < N_; ++i) steppers.emplace_back(std::size_t(nx_), std::size_t(nu_));
        std::vector<double> xstep(nx_);
        const auto integrate_i = [&](int i, std::span<const double> xi, std::span<const double> ui,
                                     Mat& Ai, Mat& Bi, bool want_sens = true) {
            return steppers[std::size_t(i)].step(dyn_, xi, ui, t_[i], t_[i + 1], opt.nsub,
                                                 xstep, Ai, Bi, want_sens);
        };

        int it = 0;
        sol.status = 1;
        for (; it <= opt.max_iter; ++it) {
            // 1. linearize at (x, u)
            for (int i = 0; i < N_; ++i) {
                integrate_i(i, x[i], u[i], A[i], B[i]);
                b[i].resize(nx_);
                for (int k = 0; k < nx_; ++k) b[i][k] = xstep[k] - x[i + 1][k];
                Mat Q(nx_, nx_), R(nu_, nu_), Sm(nu_, nx_);
                q[i].assign(nx_, 0.0); r[i].assign(nu_, 0.0);
                cost_.stage_hess(t_[i], x[i], u[i], Q, R, Sm);
                cost_.stage_grad(t_[i], x[i], u[i], q[i], r[i]);
                Qr[i] = std::move(Q); Rr[i] = std::move(R); S[i] = std::move(Sm);
            }
            q[N_].assign(nx_, 0.0);
            { Mat QN(nx_, nx_); cost_.term_hess(x[N_], QN); QNr = std::move(QN); }
            cost_.term_grad(x[N_], q[N_]);

            // 1b. exact Hessian: add the dynamics curvature Σ_k π_k ∂²Φ_k/∂(x,u)²
            //     via central differences of the adjoint gradient [Aᵀπ; Bᵀπ]
            //     (π = carried duals; zero at iteration 0 ⇒ pure Gauss-Newton there)
            if (opt.hessian == HessianMode::exact_fd && res_stat_prev < opt.exact_switch_stat) {
                curvature_on = true;
                // multipliers: the carried QP duals (lag one accepted step; empirically
                // better-conditioned curvature than the bare adjoint-sweep costates)
                const int nz = nx_ + nu_;
                Mat C(nz, nz), Ap(nx_, nx_), Bp(nx_, nu_);
                std::vector<double> zx(nx_), zu(nu_), gp(nz), gm(nz);
                for (int i = 0; i < N_; ++i) {
                    const std::vector<double>& pic = pi[i];
                    double pin = 0.0;
                    for (int k = 0; k < nx_; ++k) pin = std::max(pin, std::abs(pic[k]));
                    if (pin == 0.0) continue;
                    C.set_zero();
                    bool ok_fd = true;
                    const auto adj_grad = [&](std::span<double> g) -> bool {
                        if (!integrate_i(i, zx, zu, Ap, Bp)) return false;
                        for (int a = 0; a < nx_; ++a) {
                            double s = 0;
                            for (int p = 0; p < nx_; ++p) s += Ap(p, a) * pic[p];
                            g[a] = s;
                        }
                        for (int c = 0; c < nu_; ++c) {
                            double s = 0;
                            for (int p = 0; p < nx_; ++p) s += Bp(p, c) * pic[p];
                            g[nx_ + c] = s;
                        }
                        return true;
                    };
                    for (int j = 0; j < nz && ok_fd; ++j) {
                        const double zj = (j < nx_) ? x[i][j] : u[i][j - nx_];
                        const double eps = opt.fd_hess_eps * std::max(1.0, std::abs(zj));
                        zx.assign(x[i].begin(), x[i].end());
                        zu.assign(u[i].begin(), u[i].end());
                        (j < nx_ ? zx[j] : zu[j - nx_]) = zj + eps;
                        ok_fd = adj_grad(gp);
                        (j < nx_ ? zx[j] : zu[j - nx_]) = zj - eps;
                        ok_fd = ok_fd && adj_grad(gm);
                        if (ok_fd)
                            for (int a = 0; a < nz; ++a) C(a, j) = (gp[a] - gm[a]) / (2.0 * eps);
                    }
                    if (!ok_fd) continue;                         // graceful GN fallback per stage
                    for (int a = 0; a < nz; ++a)                  // symmetrize
                        for (int c = a + 1; c < nz; ++c) {
                            const double v = 0.5 * (C(a, c) + C(c, a));
                            C(a, c) = C(c, a) = v;
                        }
                    for (int a = 0; a < nx_; ++a)
                        for (int c = 0; c < nx_; ++c) Qr[i](a, c) += C(a, c);
                    for (int a = 0; a < nu_; ++a)
                        for (int c = 0; c < nu_; ++c) Rr[i](a, c) += C(nx_ + a, nx_ + c);
                    for (int a = 0; a < nu_; ++a)
                        for (int c = 0; c < nx_; ++c) S[i](a, c) += C(nx_ + a, c);
                }
            }

            // 2. Levenberg-Marquardt term (acados ocp_nlp_add_levenberg_marquardt_term)
            double lm;
            if (opt.adaptive_lm) {
                if (it == 0)                 { mu = mu_bar = opt.lm_mu0; }
                else if (prev_alpha == 1.0)  { double t = mu; mu = std::max(opt.lm_mu_min, mu_bar / opt.lm_lam); mu_bar = t; }
                else                         { mu = std::min(opt.lm_lam * mu, 1.0); }
                lm = opt.lm_obj_scalar * cost * mu;
            } else {
                lm = opt.lm_fixed;
            }
            for (int i = 0; i < N_; ++i) {
                for (int k = 0; k < nx_; ++k) Qr[i](k, k) += lm_scale(i) * lm;
                for (int k = 0; k < nu_; ++k) Rr[i](k, k) += lm_scale(i) * lm;
            }
            for (int k = 0; k < nx_; ++k) QNr(k, k) += lm_scale(N_) * lm;

            // NOTE: the exact stage-block Hessian may be indefinite even at the
            // optimum (only the Riccati-reduced Hessian must be PSD) — it is
            // handed to hpipm RAW; projection happens only if the QP fails (below).

            // 3. NLP residuals with the carried duals (acados ocp_nlp_res_compute):
            //    u-block i: r + Bᵀπᵢ;  x-block i: q + Aᵀπᵢ − πᵢ₋₁ (− λ_lb + λ_ub at i = 0)
            double res_stat = 0.0, res_eq = 0.0;
            for (int i = 0; i < N_; ++i) {
                for (int a = 0; a < nu_; ++a) {
                    double s = r[i][a];
                    for (int p = 0; p < nx_; ++p) s += B[i](p, a) * pi[i][p];
                    res_stat = std::max(res_stat, std::abs(s));
                }
                for (int k = 0; k < nx_; ++k) {
                    double s = q[i][k];
                    for (int p = 0; p < nx_; ++p) s += A[i](p, k) * pi[i][p];
                    if (i > 0)  s -= pi[i - 1][k];
                    else        s -= lam_lb0[k] - lam_ub0[k];
                    res_stat = std::max(res_stat, std::abs(s));
                }
                for (int k = 0; k < nx_; ++k) res_eq = std::max(res_eq, std::abs(b[i][k]));
            }
            for (int k = 0; k < nx_; ++k)
                res_stat = std::max(res_stat, std::abs(q[N_][k] - pi[N_ - 1][k]));

            sol.stats.push_back({res_stat, res_eq, cost, lm, 0.0, step_norm, -1, 0});
            sol.statio = res_stat; sol.res_eq = res_eq;
            res_stat_prev = res_stat;
            if (it == 0) {
                infeasible_guess = !(res_eq <= opt.tol_eq);
                double l1 = 0.0;
                for (int i = 0; i < N_; ++i) for (int k = 0; k < nx_; ++k) l1 += std::abs(b[i][k]);
                glob_.restart(l1);
            }
            if (opt.verbose)
                std::printf("  it %3d  cost %.8e  res_stat %.3e  res_eq %.3e  lm %.2e  step %.2e\n",
                            it, cost, res_stat, res_eq, lm, step_norm);

            // 4. termination (acados check_termination order)
            if (std::isnan(res_stat) || std::isnan(res_eq)) { sol.status = 4; break; }
            if (res_eq < opt.tol_eq) {
                if (res_stat < opt.tol_stat)                            { sol.status = 0; break; }
                if (opt.adaptive_lm && cost < opt.tol_zero_res)         { sol.status = 0; break; }
            }
            if (it > 0 && step_norm < opt.tol_eq)                       { sol.status = 5; break; }
            if (it >= opt.max_iter)                                     { sol.status = 1; break; }

            // 5. backward pass (hpipm Riccati QP)
            const auto assemble_qp = [&] {
                for (int i = 0; i < N_; ++i) {
                    auto cA = colmajor(A[i]), cB = colmajor(B[i]),
                         cQ = colmajor(Qr[i]), cR = colmajor(Rr[i]), cS = colmajor(S[i]);
                    qp.setA(i, cA.data()); qp.setB(i, cB.data()); qp.setb(i, b[i].data());
                    qp.setQ(i, cQ.data()); qp.setR(i, cR.data()); qp.setS(i, cS.data());
                    qp.setq(i, q[i].data()); qp.setr(i, r[i].data());
                }
                auto cQN = colmajor(QNr); qp.setQ(N_, cQN.data()); qp.setq(N_, q[N_].data());
                for (int k = 0; k < nx_; ++k) x0bound[k] = x0[k] - x[0][k];
                qp.setx0(x0bound.data());
            };
            assemble_qp();
            int qs = qp.solve();
            if (qs != 0 && qs != 1 && curvature_on) {
                // indefinite exact Hessian broke the Riccati/IPM: project each
                // stage block onto the PSD cone and retry once (on-demand reg)
                const int nz = nx_ + nu_;
                Mat W(nz, nz);
                for (int i = 0; i < N_; ++i) {
                    for (int a = 0; a < nx_; ++a)
                        for (int c = 0; c < nx_; ++c) W(a, c) = Qr[i](a, c);
                    for (int a = 0; a < nu_; ++a)
                        for (int c = 0; c < nu_; ++c) W(nx_ + a, nx_ + c) = Rr[i](a, c);
                    for (int a = 0; a < nu_; ++a)
                        for (int c = 0; c < nx_; ++c) { W(nx_ + a, c) = S[i](a, c); W(c, nx_ + a) = S[i](a, c); }
                    project_psd(W, opt.reg_min_eig, opt.reg_mirror);
                    for (int a = 0; a < nx_; ++a)
                        for (int c = 0; c < nx_; ++c) Qr[i](a, c) = W(a, c);
                    for (int a = 0; a < nu_; ++a)
                        for (int c = 0; c < nu_; ++c) Rr[i](a, c) = W(nx_ + a, nx_ + c);
                    for (int a = 0; a < nu_; ++a)
                        for (int c = 0; c < nx_; ++c) S[i](a, c) = W(nx_ + a, c);
                }
                project_psd(QNr, opt.reg_min_eig, opt.reg_mirror);
                assemble_qp();
                qs = qp.solve();
            }
            curvature_on = false;
            sol.stats.back().qp_status = qs; sol.stats.back().qp_iter = qp.iter();
            if (qs != 0 && qs != 1) { sol.status = 2; break; }   // acados tolerates QP maxiter (1)

            for (int i = 0; i <= N_; ++i) { dx[i].assign(nx_, 0.0); qp.getx(i, dx[i].data()); }
            for (int i = 0; i < N_; ++i) {
                du[i].assign(nu_, 0.0);          qp.getu(i, du[i].data());
                Kc[i].assign(nu_ * nx_, 0.0);    qp.getK(i, Kc[i].data());
                kc[i].assign(nu_, 0.0);          qp.getk(i, kc[i].data());
                pi_qp[i].assign(nx_, 0.0);       qp.getpi(i, pi_qp[i].data());
            }
            qp.getlam_lbx0(lamlb_qp.data()); qp.getlam_ubx0(lamub_qp.data());

            // step norm ‖d‖∞ and predicted reductions
            step_norm = 0.0;
            for (int i = 0; i <= N_; ++i) for (double v : dx[i]) step_norm = std::max(step_norm, std::abs(v));
            for (int i = 0; i < N_; ++i)  for (double v : du[i]) step_norm = std::max(step_norm, std::abs(v));
            double qp_obj = 0.0, gd = 0.0;
            for (int i = 0; i < N_; ++i) {
                for (int k = 0; k < nx_; ++k) { double s = 0; for (int l = 0; l < nx_; ++l) s += Qr[i](k, l) * dx[i][l]; qp_obj += 0.5 * dx[i][k] * s; }
                for (int a = 0; a < nu_; ++a) { double s = 0; for (int c = 0; c < nu_; ++c) s += Rr[i](a, c) * du[i][c]; qp_obj += 0.5 * du[i][a] * s; }
                for (int a = 0; a < nu_; ++a) { double s = 0; for (int k = 0; k < nx_; ++k) s += S[i](a, k) * dx[i][k]; qp_obj += du[i][a] * s; }
                for (int k = 0; k < nx_; ++k) gd += q[i][k] * dx[i][k];
                for (int a = 0; a < nu_; ++a) gd += r[i][a] * du[i][a];
            }
            for (int k = 0; k < nx_; ++k) { double s = 0; for (int l = 0; l < nx_; ++l) s += QNr(k, l) * dx[N_][l]; qp_obj += 0.5 * dx[N_][k] * s; }
            for (int k = 0; k < nx_; ++k) gd += q[N_][k] * dx[N_][k];
            qp_obj += gd;
            const double pred_qp = -qp_obj, pred_lin = -gd;

            // 6. forward pass: feasible rollout with globalization
            std::vector<std::vector<double>> xn, un;
            double cn = 0, alpha;
            auto rollout = [&](double a) -> bool {
                xn.assign(N_ + 1, {}); un.assign(N_, {});
                xn[0].resize(nx_);
                for (int k = 0; k < nx_; ++k) xn[0][k] = x[0][k] + a * dx[0][k];
                bool ok = true;
                Mat At(nx_, nx_), Bt(nx_, nu_);
                for (int i = 0; i < N_; ++i) {
                    un[i].resize(nu_);
                    for (int c = 0; c < nu_; ++c) {
                        double s = a * kc[i][c];
                        for (int k = 0; k < nx_; ++k) s += Kc[i][c + k * nu_] * (xn[i][k] - x[i][k]);
                        un[i][c] = u[i][c] + s;
                    }
                    if (!integrate_i(i, xn[i], un[i], At, Bt, /*want_sens=*/false)) ok = false;
                    xn[i + 1] = xstep;                // NaN-filled on failure; keep propagating
                }
                cn = ok ? total_cost(xn, un) : nan;
                return ok;
            };

            if (infeasible_guess) {
                // acados: forced full step to obtain a feasible iterate (no acceptance test)
                alpha = 1.0;
                rollout(alpha);
                infeasible_guess = false;
            } else {
                alpha = glob_.alpha0();
                bool accepted = false;
                while (true) {
                    rollout(alpha);
                    if (glob_.accept(cost, cn, pred_qp, pred_lin, alpha, 0.0, 0.0)) { accepted = true; break; }
                    alpha *= opt.alpha_reduction;
                    if (alpha < opt.alpha_min) break;
                }
                if (!accepted) { sol.status = 3; break; }
            }
            x = std::move(xn); u = std::move(un); cost = cn;
            pi = pi_qp; lam_lb0 = lamlb_qp; lam_ub0 = lamub_qp;   // full-step dual update
            prev_alpha = alpha;
            sol.stats.back().alpha = alpha;
        }

        // exact adjoint costates at the final iterate (dual seed; every exit path
        // leaves q, A linearized at the final (x, u))
        std::vector<std::vector<double>> lam(N_ + 1);
        lam[N_] = q[N_].empty() ? std::vector<double>(nx_, 0.0) : q[N_];
        for (int i = N_ - 1; i >= 0; --i) {
            lam[i] = q[i];
            for (int k = 0; k < nx_; ++k) {
                double s = 0;
                for (int p = 0; p < nx_; ++p) s += A[i](p, k) * lam[i + 1][p];
                lam[i][k] += s;
            }
        }

        sol.x = std::move(x); sol.u = std::move(u); sol.costate = std::move(lam);
        sol.cost = cost; sol.iters = it;
        return sol;
    }

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
