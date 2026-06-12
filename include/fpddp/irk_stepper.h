#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · irk_stepper.h
//
//  Persistent fixed-step IRK stepper (Gauss-Legendre, default s=3) with inline
//  exact discrete sensitivities and a hardened Newton:
//
//    * the Butcher tableau is built once and cached (it was 52% of the old
//      per-call cost), all workspace lives in the stepper across calls;
//    * stage solve escalation:  simplified Newton (frozen J, contraction
//      monitored — cheap, common case)  →  cold-predictor retry  →  full
//      Newton on the collocation system (per-stage Jacobians refreshed every
//      iteration — the acados sim_irk scheme, but convergence-checked)  →
//      substep halving (up to max_splits)  →  loud failure (ok = false,
//      NaN state).  Unlike acados' sim_irk, an unconverged stage system is
//      NEVER returned as success;
//    * stage displacements of the first substep are warm-started from the
//      previous call (the repeated-linearization pattern of DDP), later
//      substeps are predicted by collocation extrapolation;
//    * sensitivities are the exact variational derivatives of the realized
//      map: per-stage Jacobians at the converged stages, fresh factorization,
//      chained across substeps (the scheme validated in test/sens_test.cpp).
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "fpddp/model.h"
#include "irk/essence/collocation.hpp"
#include "irk/notion/stages.hpp"
#include "irk/being.hpp"

namespace fpddp {

// Adapts {dynamics, control held constant} to irk's ControlledField concept.
struct ControlledField {
    const AnyDynamics& dyn;
    std::span<const double> u;
    std::size_t m;
    void operator()(double t, std::span<const double> x, std::span<double> dx) const { dyn.rhs(t, x, u, dx); }
    void jacobian(double t, std::span<const double> x, Mat& J) const { dyn.fx(t, x, u, J); }
    void control_jacobian(double t, std::span<const double> x, Mat& B) const { dyn.fu(t, x, u, B); }
    std::size_t control_dim() const { return m; }
};

// process-wide tableau cache
inline const irk::essence::ButcherTableau<double>& irk_tableau(irk::essence::Family fam, std::size_t s) {
    static std::map<std::pair<int, std::size_t>, std::unique_ptr<const irk::essence::ButcherTableau<double>>> cache;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto& slot = cache[{int(fam), s}];
    if (!slot)
        slot = std::make_unique<const irk::essence::ButcherTableau<double>>(
            irk::essence::ButcherTableau<double>::make(fam, s));
    return *slot;
}

struct IrkStats {
    std::size_t leaves = 0;          // substeps actually taken (≥ nsub when splitting)
    std::size_t newton_iters = 0;    // simplified-Newton iterations
    std::size_t full_newton = 0;     // full-Newton activations
    std::size_t splits = 0;          // substep halvings
    bool warm_hit = false;           // first leaf converged from the warm start
};

class IrkStepper {
public:
    IrkStepper(std::size_t nx, std::size_t nu,
               irk::essence::Family fam = irk::essence::Family::gauss_legendre, std::size_t s = 3)
        : n_(nx), m_(nu), s_(s), tab_(&irk_tableau(fam, s)), solver_(nx),
          Jk_(s, Mat(nx, nx)), Bck_(s, Mat(nx, nu)),
          As_(nx, nx), Bs_(nx, nu), A_acc_(nx, nx), B_acc_(nx, nu), AB_tmp_(nx, nx > nu ? nx : nu),
          x_cur_(nx), Yk_(nx), Z_(s * nx), Zwarm_(s * nx), rates_(s * nx), rhs_(s * nx) {}

    // knobs (defaults give the hardened behavior; max_splits = 0 disables splitting)
    std::size_t newton_max = 10, full_newton_max = 12, max_splits = 3;
    double rtol = 1e-12, atol = 1e-12;

    void forget() { have_warm_ = false; }     // drop cross-call warm-start state

    // integrate [t0,t1] with u constant; fills xnext (n), A = ∂x⁺/∂x, B = ∂x⁺/∂u.
    // Returns false (and NaN-fills xnext) when even the escalation chain fails.
    // want_sens = false skips the variational solves (pure rollout — line search).
    bool step(const AnyDynamics& dyn, std::span<const double> x, std::span<const double> u,
              double t0, double t1, std::size_t nsub,
              std::vector<double>& xnext, Mat& A, Mat& B, bool want_sens = true) {
        ControlledField f{dyn, u, m_};
        stats_ = {};
        want_sens_ = want_sens;
        meas_ = irk::being::Measure<double>{rtol, atol};
        for (std::size_t i = 0; i < n_; ++i) x_cur_[i] = x[i];
        set_identity(A_acc_); B_acc_.set_zero();
        prev_valid_ = false;
        first_leaf_ = true;

        const double h = (t1 - t0) / double(nsub);
        bool ok = true;
        for (std::size_t k = 0; k < nsub && ok; ++k)
            ok = advance(f, t0 + double(k) * h, h, 0);

        if (!ok) {
            xnext.assign(n_, std::numeric_limits<double>::quiet_NaN());
            A = Mat::identity(n_); B = Mat(n_, m_);
            have_warm_ = false;
            return false;
        }
        xnext.assign(x_cur_.begin(), x_cur_.end());
        A = A_acc_; B = B_acc_;
        return true;
    }

    [[nodiscard]] const IrkStats& stats() const noexcept { return stats_; }

private:
    // one collocation step of size h from (t, x_cur_); recurses by halving on failure
    bool advance(const ControlledField& f, double t, double h, std::size_t depth) {
        const bool tried_warm = predict(t, h);

        (void)solver_.refresh_jacobian(f, t, std::span<const double>(x_cur_));
        auto rep = solver_.solve(f, *tab_, t, std::span<const double>(x_cur_), h,
                                 std::span<double>(Z_), meas_,
                                 irk::notion::NewtonControls{newton_max, 1e-2});
        stats_.newton_iters += rep.iterations;
        bool ok = rep.converged();

        if (!ok && tried_warm) {                          // cold retry: extrapolated/zero predictor
            predict_cold(t, h);
            rep = solver_.solve(f, *tab_, t, std::span<const double>(x_cur_), h,
                                std::span<double>(Z_), meas_,
                                irk::notion::NewtonControls{newton_max, 1e-2});
            stats_.newton_iters += rep.iterations;
            ok = rep.converged();
        }
        if (!ok) {                                        // full Newton on the collocation system
            ++stats_.full_newton;
            ok = full_newton(f, t, h);
        }
        if (!ok) {
            if (depth < max_splits) {
                ++stats_.splits;
                return advance(f, t, 0.5 * h, depth + 1) && advance(f, t + 0.5 * h, 0.5 * h, depth + 1);
            }
            return false;
        }
        if (tried_warm && first_leaf_) stats_.warm_hit = true;

        if (want_sens_ && !leaf_sensitivity(f, t, h)) return false;  // singular variational matrix: loud

        // stash warm start (first leaf) and the node for in-call extrapolation
        if (first_leaf_) { Zwarm_ = Z_; warm_h_ = h; have_warm_ = true; first_leaf_ = false; }
        prev_y_.assign(x_cur_.begin(), x_cur_.end());
        prev_Z_ = Z_; prev_t_ = t; prev_h_ = h; prev_valid_ = true;

        for (std::size_t i = 0; i < s_; ++i)              // x⁺ = x + Σ b̃ᵢ Zᵢ
            for (std::size_t a = 0; a < n_; ++a) x_cur_[a] += tab_->b_tilde[i] * Z_[i * n_ + a];
        ++stats_.leaves;
        return true;
    }

    // returns true if the warm start was used
    bool predict(double t, double h) {
        if (first_leaf_ && have_warm_ && std::abs(h - warm_h_) <= 1e-12 * std::abs(h)) {
            const double zn = meas_.norm(std::span<const double>(Zwarm_), std::span<const double>(x_cur_));
            if (std::isfinite(zn) && zn < 1e5) { Z_ = Zwarm_; return true; }
        }
        predict_cold(t, h);
        return false;
    }

    // collocation extrapolation from the previous leaf (zero on the first)
    void predict_cold(double t, double h) {
        if (!prev_valid_) { for (auto& z : Z_) z = 0.0; return; }
        for (std::size_t i = 0; i < s_; ++i) {
            const double sigma = (t + tab_->c[i] * h - prev_t_) / prev_h_;
            const auto w = tab_->output_weights(sigma);
            for (std::size_t a = 0; a < n_; ++a) {
                double acc = prev_y_[a] - x_cur_[a];
                for (std::size_t j = 0; j < s_; ++j) acc += w[j] * prev_Z_[j * n_ + a];
                Z_[i * n_ + a] = acc;
            }
        }
        const double zn = meas_.norm(std::span<const double>(Z_), std::span<const double>(x_cur_));
        if (!std::isfinite(zn) || zn > 1e5)
            for (auto& z : Z_) z = 0.0;
    }

    // full Newton on G(Z) = 0 with per-stage Jacobians refreshed every iteration;
    // convergence judged in the scaled norm (loud failure — never silent)
    bool full_newton(const ControlledField& f, double t, double h) {
        constexpr double eps = std::numeric_limits<double>::epsilon();
        const double kappa = std::max(1e-2, 10.0 * eps / rtol);
        double previous = 0.0;
        for (std::size_t it = 0; it < full_newton_max; ++it) {
            for (std::size_t k = 0; k < s_; ++k) {        // J_k, f(Y_k) at current stages
                for (std::size_t a = 0; a < n_; ++a) Yk_[a] = x_cur_[a] + Z_[k * n_ + a];
                Jk_[k].set_zero();
                f.jacobian(t + tab_->c[k] * h, std::span<const double>(Yk_), Jk_[k]);
                f(t + tab_->c[k] * h, std::span<const double>(Yk_),
                  std::span<double>(rates_).subspan(k * n_, n_));
            }
            auto lu = factor_variational(h);
            if (!lu) return false;
            for (std::size_t i = 0; i < s_; ++i)          // rhs = −G = −Z + h Σ a_ik f(Y_k)
                for (std::size_t a = 0; a < n_; ++a) {
                    double acc = -Z_[i * n_ + a];
                    for (std::size_t k = 0; k < s_; ++k)
                        acc += h * tab_->A(i, k) * rates_[k * n_ + a];
                    rhs_[i * n_ + a] = acc;
                }
            lu->solve(std::span<double>(rhs_));
            const double dn = meas_.norm(std::span<const double>(rhs_), std::span<const double>(x_cur_));
            if (!std::isfinite(dn)) return false;
            for (std::size_t idx = 0; idx < s_ * n_; ++idx) Z_[idx] += rhs_[idx];
            if (dn <= 1e-2 * kappa) return true;
            if (it > 0) {
                const double theta = dn / previous;
                if (theta < 0.5 && theta / (1.0 - theta) * dn <= kappa) return true;
                if (!(theta < 10.0) && dn > kappa) return false;     // diverging
            }
            previous = dn;
        }
        return false;
    }

    // exact variational matrix M = I − h(A ⊗ J_k) at the current stage values
    [[nodiscard]] irk::expected<irk::essence::PLU<double>, irk::essence::Singular>
    factor_variational(double h) {
        const std::size_t sn = s_ * n_;
        Mat M(sn, sn);
        for (std::size_t i = 0; i < s_; ++i)
            for (std::size_t k = 0; k < s_; ++k) {
                const double w = -h * tab_->A(i, k);
                for (std::size_t a = 0; a < n_; ++a)
                    for (std::size_t b = 0; b < n_; ++b)
                        M(i * n_ + a, k * n_ + b) = w * Jk_[k](a, b);
            }
        for (std::size_t d = 0; d < sn; ++d) M(d, d) += 1.0;
        return irk::essence::PLU<double>::factor(std::move(M));
    }

    // sensitivities of the realized leaf map (per-stage J/Bc at the solution),
    // chained into A_acc_/B_acc_:  A ← As·A,  B ← As·B + Bs
    bool leaf_sensitivity(const ControlledField& f, double t, double h) {
        for (std::size_t k = 0; k < s_; ++k) {
            for (std::size_t a = 0; a < n_; ++a) Yk_[a] = x_cur_[a] + Z_[k * n_ + a];
            Jk_[k].set_zero();  f.jacobian(t + tab_->c[k] * h, std::span<const double>(Yk_), Jk_[k]);
            Bck_[k].set_zero(); f.control_jacobian(t + tab_->c[k] * h, std::span<const double>(Yk_), Bck_[k]);
        }
        auto lu = factor_variational(h);
        if (!lu) return false;

        set_identity(As_);
        for (std::size_t a = 0; a < n_; ++a) {            // columns of dZ/dx
            for (std::size_t i = 0; i < s_; ++i)
                for (std::size_t p = 0; p < n_; ++p) {
                    double acc = 0;
                    for (std::size_t k = 0; k < s_; ++k) acc += tab_->A(i, k) * Jk_[k](p, a);
                    rhs_[i * n_ + p] = h * acc;
                }
            lu->solve(std::span<double>(rhs_));
            for (std::size_t p = 0; p < n_; ++p) {
                double acc = 0;
                for (std::size_t i = 0; i < s_; ++i) acc += tab_->b_tilde[i] * rhs_[i * n_ + p];
                As_(p, a) += acc;
            }
        }
        Bs_.set_zero();
        for (std::size_t c = 0; c < m_; ++c) {            // columns of dZ/du
            for (std::size_t i = 0; i < s_; ++i)
                for (std::size_t p = 0; p < n_; ++p) {
                    double acc = 0;
                    for (std::size_t k = 0; k < s_; ++k) acc += tab_->A(i, k) * Bck_[k](p, c);
                    rhs_[i * n_ + p] = h * acc;
                }
            lu->solve(std::span<double>(rhs_));
            for (std::size_t p = 0; p < n_; ++p) {
                double acc = 0;
                for (std::size_t i = 0; i < s_; ++i) acc += tab_->b_tilde[i] * rhs_[i * n_ + p];
                Bs_(p, c) += acc;
            }
        }
        // chain
        mat_mul(As_, A_acc_, AB_tmp_, n_);  copy_into(AB_tmp_, A_acc_, n_, n_);
        mat_mul(As_, B_acc_, AB_tmp_, m_);
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < m_; ++j) B_acc_(i, j) = AB_tmp_(i, j) + Bs_(i, j);
        return true;
    }

    void set_identity(Mat& M) { M.set_zero(); for (std::size_t k = 0; k < n_; ++k) M(k, k) = 1.0; }
    // O = L·R into the (n_ × cols) top-left of O
    void mat_mul(const Mat& L, const Mat& R, Mat& O, std::size_t cols) {
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < cols; ++j) {
                double acc = 0;
                for (std::size_t k = 0; k < n_; ++k) acc += L(i, k) * R(k, j);
                O(i, j) = acc;
            }
    }
    void copy_into(const Mat& S, Mat& D, std::size_t r, std::size_t c) {
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < c; ++j) D(i, j) = S(i, j);
    }

    std::size_t n_, m_, s_;
    const irk::essence::ButcherTableau<double>* tab_;
    irk::being::Measure<double> meas_{1e-12, 1e-12};
    irk::notion::StageSolver<double> solver_;
    std::vector<Mat> Jk_, Bck_;
    Mat As_, Bs_, A_acc_, B_acc_, AB_tmp_;
    std::vector<double> x_cur_, Yk_, Z_, Zwarm_, rates_, rhs_;
    std::vector<double> prev_y_, prev_Z_;
    double prev_t_ = 0, prev_h_ = 0, warm_h_ = 0;
    bool prev_valid_ = false, have_warm_ = false, first_leaf_ = true, want_sens_ = true;
    IrkStats stats_{};
};

}  // namespace fpddp
