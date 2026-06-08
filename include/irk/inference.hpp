#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · inference.hpp                                                [layer 5]
//
//  INFERENCE (Schluss; Chelpanov: умозаключение).  The syllogism unites the
//  universal (the method — a tableau), the particular (the problem — a vector
//  field in its present state) and concludes the singular (the trajectory).
//  Integration is that inference performed in time: every step proposes a
//  conclusion, submits it to judgment, and either enters it into the record
//  or withdraws and retries with a humbler premise.
//
//  Exports:   Stats, Options, Node, Solution, Breakdown, Failure, integrate.
//  Includes:  being, essence/collocation, notion/stages, judgment.
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "irk/expected.hpp"
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "irk/being.hpp"
#include "irk/essence/collocation.hpp"
#include "irk/judgment.hpp"
#include "irk/notion/stages.hpp"
#include "irk/sensitivity.hpp"

namespace irk::inference {

struct Stats {
    std::size_t nfev{0};            // vector-field evaluations
    std::size_t njac{0};            // Jacobian refreshes
    std::size_t nlu{0};             // LU factorizations (stage + filter)
    std::size_t naccept{0};         // accepted steps
    std::size_t nreject_error{0};   // rejections by the error judgment
    std::size_t nreject_newton{0};  // rejections by Newton breakdown
    std::size_t niter{0};           // total Newton iterations
    std::map<std::size_t, std::size_t> accepted_by_stage;
};

template <being::Real R>
struct Options {
    essence::Family family{essence::Family::radau_iia};
    std::vector<std::size_t> stage_ladder{2, 3, 5, 7};  // rungs of the order ladder
    std::optional<R> first_step{};
    R max_step{std::numeric_limits<R>::infinity()};
    std::size_t max_steps{200000};                      // budget of step *attempts*
    std::optional<R> fixed_step{};                      // disables adaptivity
    std::optional<std::size_t> fixed_stages{};          // pins the stage count
    judgment::StepControls<R> step{};
    judgment::OrderControls order{};
    notion::NewtonControls newton{};
};

// One accepted step — everything needed to reproduce its collocation polynomial.
template <being::Real R>
struct Node {
    R t{};                // departure time
    R hs{};               // signed step taken
    std::vector<R> y;     // departure state
    std::vector<R> Z;     // stage displacements, stage-major
    std::shared_ptr<const essence::ButcherTableau<R>> tableau;
};

enum class Breakdown { step_underflow, max_steps, newton_failure };

[[nodiscard]] constexpr const char* name(Breakdown b) noexcept {
    switch (b) {
        case Breakdown::step_underflow: return "step underflow";
        case Breakdown::max_steps:      return "attempt budget exhausted";
        case Breakdown::newton_failure: return "Newton failure at fixed step";
    }
    return "?";
}

namespace detail {
template <typename R, typename F> class Integrator;
}

template <being::Real R>
class Solution {
public:
    Solution() = default;

    [[nodiscard]] R t_begin() const noexcept { return t0_; }
    [[nodiscard]] R t_end()   const noexcept { return t1_; }
    [[nodiscard]] std::span<const R> y_end() const noexcept { return y_; }
    [[nodiscard]] std::size_t steps() const noexcept { return nodes_.size(); }
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::vector<Node<R>>& nodes() const noexcept { return nodes_; }

    // Discrete forward sensitivities of the whole integrated map (populated only
    // for a ControlledField): state_sensitivity() = ∂x(t_end)/∂x(t_begin) (n×n),
    // control_sensitivity() = ∂x(t_end)/∂u (n×m, u held constant over the span).
    [[nodiscard]] bool has_sensitivity() const noexcept { return Ax_.rows() != 0; }
    [[nodiscard]] const essence::Matrix<R>& state_sensitivity()   const noexcept { return Ax_; }
    [[nodiscard]] const essence::Matrix<R>& control_sensitivity() const noexcept { return Bu_; }

    // Collocation dense output: exact at every accepted point and
    // order-consistent in between.  Times outside the integrated span are
    // clamped to the nearer endpoint.
    void eval(R t, std::span<R> out) const {
        if (nodes_.empty()) {
            for (std::size_t i = 0; i < out.size(); ++i) out[i] = y_[i];
            return;
        }
        const R dir = (t1_ >= t0_) ? R(1) : R(-1);
        std::size_t lo = 0, hi = nodes_.size() - 1;
        while (lo < hi) {                        // last node departing at/before t
            const std::size_t mid = (lo + hi + 1) / 2;
            if (dir * (t - nodes_[mid].t) >= R(0)) lo = mid; else hi = mid - 1;
        }
        const Node<R>& nd = nodes_[lo];
        const R sigma = std::clamp((t - nd.t) / nd.hs, R(0), R(1));
        const auto w  = nd.tableau->output_weights(sigma);
        const std::size_t n = nd.y.size();
        for (std::size_t i = 0; i < n; ++i) out[i] = nd.y[i];
        for (std::size_t i = 0; i < nd.tableau->s; ++i)
            being::axpy(w[i], std::span<const R>(nd.Z).subspan(i * n, n), out);
    }

    [[nodiscard]] std::vector<R> eval(R t) const {
        std::vector<R> out(y_.size());
        eval(t, std::span<R>(out));
        return out;
    }

private:
    template <typename, typename> friend class detail::Integrator;

    std::vector<Node<R>> nodes_;
    std::vector<R> y_;            // state at t1_
    R t0_{}, t1_{};
    Stats stats_{};
    essence::Matrix<R> Ax_, Bu_;  // discrete sensitivities (empty unless a ControlledField)
};

template <being::Real R>
struct Failure {
    Breakdown kind{};
    R t{};                        // time reached
    Solution<R> partial;          // trajectory up to t; its stats are complete

    [[nodiscard]] const Stats& stats() const noexcept { return partial.stats(); }
};

namespace detail {

template <typename R, typename F>
class Integrator {
public:
    Integrator(const F& field, being::Interval<R> iv, std::span<const R> y0,
               being::Measure<R> meas, Options<R> opt)
        : f_(field), iv_(iv), meas_(meas), opt_(std::move(opt)),
          n_(y0.size()), y_(y0.begin(), y0.end()), solver_(y0.size()) {
        auto ladder = opt_.fixed_stages ? std::vector<std::size_t>{*opt_.fixed_stages}
                                        : opt_.stage_ladder;
        std::ranges::sort(ladder);
        ladder.erase(std::unique(ladder.begin(), ladder.end()), ladder.end());
        if (ladder.empty())
            throw std::invalid_argument("irk: the stage ladder must not be empty");
        tabs_.reserve(ladder.size());
        for (const std::size_t s : ladder)
            tabs_.push_back(std::make_shared<const essence::ButcherTableau<R>>(
                essence::ButcherTableau<R>::make(opt_.family, s)));
    }

    [[nodiscard]] irk::expected<Solution<R>, Failure<R>> run() {
        sol_.t0_ = iv_.t0;
        t_       = iv_.t0;
        if (iv_.t1 == iv_.t0) return finish();

        const R dir = iv_.direction();
        constexpr R eps = std::numeric_limits<R>::epsilon();

        std::vector<R> f0(n_);                   // opening rate, for the first judgment
        f_(t_, std::span<const R>(y_), std::span<R>(f0));
        ++st_.nfev;

        judgment::OrderJudgment oj(tabs_.size());
        const bool fixed = opt_.fixed_step.has_value();
        R h;
        if (fixed)                h = std::abs(*opt_.fixed_step);
        else if (opt_.first_step) h = std::abs(*opt_.first_step);
        else                      h = judgment::initial_step<R>(
                                          f_, t_, y_, f0, dir,
                                          tabs_[oj.rung()]->controller_order, meas_,
                                          opt_.max_step, iv_.length(), st_.nfev);
        h = std::min({h, opt_.max_step, iv_.length()});

        bool need_jacobian = true;
        bool rejected      = false;
        std::size_t newton_failures = 0;
        std::size_t attempts        = 0;
        std::vector<R> ynew(n_), est(n_);

        while (dir * (iv_.t1 - t_) > R(0)) {
            if (++attempts > opt_.max_steps) return fail(Breakdown::max_steps);

            const R hmin      = R(16) * eps * std::max(std::abs(t_), R(1));
            const R remaining = iv_.t1 - t_;
            R hs;
            if (fixed) hs = (std::abs(remaining) <= h * (R(1) + R(1e-8))) ? remaining
                                                                          : dir * h;
            else       hs = (std::abs(remaining) <  R(1.1) * h) ? remaining : dir * h;
            if (std::abs(hs) < hmin) return fail(Breakdown::step_underflow);

            const auto& tab = *tabs_[oj.rung()];
            const std::size_t s = tab.s;

            if (need_jacobian) {
                st_.nfev += solver_.refresh_jacobian(f_, t_, std::span<const R>(y_));
                ++st_.njac;
                need_jacobian = false;
            }

            Z_.assign(s * n_, R(0));
            predict(tab, hs);

            const auto rep = solver_.solve(f_, tab, t_, std::span<const R>(y_), hs,
                                           std::span<R>(Z_), meas_, opt_.newton);
            st_.nfev  += rep.nfev;
            st_.nlu   += rep.factorizations;
            st_.niter += rep.iterations;

            if (!rep.converged()) {              // Newton broke down: shrink, maybe demote
                ++st_.nreject_newton;
                if (fixed) return fail(Breakdown::newton_failure);
                ++newton_failures;
                rejected = true;
                if (newton_failures >= 2 && oj.rung() > 0) {
                    oj.demote();
                    newton_failures = 0;
                }
                h *= R(0.5);
                continue;                        // same point: Jacobian reused
            }

            ynew = y_;                           // y₁ = y₀ + Σ b̃ᵢ Zᵢ
            for (std::size_t i = 0; i < s; ++i)
                being::axpy(tab.b_tilde[i], Zblock(i), std::span<R>(ynew));

            bool accept = true;
            R factor    = R(1);
            if (!fixed) {
                for (auto& e : est) e = R(0);    // est = Σ d̃ᵢ Zᵢ  (= h Σ dᵢ f(Yᵢ))
                for (std::size_t i = 0; i < s; ++i)
                    being::axpy(tab.d_tilde[i], Zblock(i), std::span<R>(est));
                if (tab.filter_estimate) filter(hs, std::span<R>(est));
                const R err = meas_.norm2ref(est, y_, ynew);
                const auto verdict =
                    judgment::judge_step(err, tab.controller_order, opt_.step, rejected);
                accept = verdict.accept;
                factor = verdict.factor;
            }

            if (!accept) {
                ++st_.nreject_error;
                rejected = true;
                h = std::abs(hs) * factor;
                continue;                        // same point: Jacobian reused
            }

            sol_.nodes_.push_back(Node<R>{t_, hs, y_, Z_, tabs_[oj.rung()]});
            ++st_.naccept;
            ++st_.accepted_by_stage[s];

            t_ = (hs == remaining) ? iv_.t1 : t_ + hs;
            y_.swap(ynew);

            if (!fixed) {
                h = std::min(std::abs(hs) * factor, opt_.max_step);
                oj.on_accept(rep.iterations, double(rep.contraction), opt_.order,
                             opt_.newton.max_iterations);
            }
            rejected        = false;
            newton_failures = 0;
            need_jacobian   = true;
        }
        return finish();
    }

private:
    [[nodiscard]] std::span<const R> Zblock(std::size_t i) const noexcept {
        return std::span<const R>(Z_).subspan(i * n_, n_);
    }

    // Extrapolate the previous collocation polynomial as the Newton predictor;
    // distrust it wholesale if it is wild.
    void predict(const essence::ButcherTableau<R>& tab, R hs) {
        if (sol_.nodes_.empty()) return;         // Z = 0
        const Node<R>& P = sol_.nodes_.back();
        for (std::size_t i = 0; i < tab.s; ++i) {
            const R sigma = (t_ + tab.c[i] * hs - P.t) / P.hs;
            const auto w  = P.tableau->output_weights(sigma);
            auto Zi = std::span<R>(Z_).subspan(i * n_, n_);
            for (std::size_t a = 0; a < n_; ++a) Zi[a] = P.y[a] - y_[a];
            for (std::size_t j = 0; j < P.tableau->s; ++j)
                being::axpy(w[j], std::span<const R>(P.Z).subspan(j * n_, n_), Zi);
        }
        const R zn = meas_.norm(Z_, y_);
        if (!std::isfinite(zn) || zn > R(1e5))
            for (auto& z : Z_) z = R(0);
    }

    // Stiff filter:  est ← (I − h J)⁻¹ est.  A singular factor leaves the raw
    // estimate in place — conservative, never fatal.
    void filter(R hs, std::span<R> est) {
        essence::Matrix<R> Fm(n_, n_);
        const auto& J = solver_.jac();
        for (std::size_t a = 0; a < n_; ++a)
            for (std::size_t b = 0; b < n_; ++b) Fm(a, b) = -hs * J(a, b);
        for (std::size_t a = 0; a < n_; ++a) Fm(a, a) += R(1);
        ++st_.nlu;
        if (auto lu = essence::PLU<R>::factor(std::move(Fm))) lu->solve(est);
    }

    // Discrete forward sensitivities of the integrated map, accumulated over the
    // accepted steps (each Node carries departure y, stage Z, tableau).  Compiled
    // away entirely for a plain VectorField.
    void compute_sensitivities() {
        if constexpr (notion::ControlledField<F, R>) {
            const std::size_t m = f_.control_dim();
            essence::Matrix<R> Ax = essence::Matrix<R>::identity(n_);
            essence::Matrix<R> Bu(n_, m);
            for (const auto& nd : sol_.nodes_) {
                essence::Matrix<R> As(n_, n_), Bs(n_, m);
                sensitivity::step_sensitivity<R>(f_, *nd.tableau, nd.t,
                    std::span<const R>(nd.y), nd.hs, std::span<const R>(nd.Z), n_, m, As, Bs);
                Ax = sensitivity::matmul(As, Ax);
                Bu = sensitivity::add(sensitivity::matmul(As, Bu), Bs);
            }
            sol_.Ax_ = std::move(Ax);
            sol_.Bu_ = std::move(Bu);
        }
    }

    [[nodiscard]] irk::expected<Solution<R>, Failure<R>> finish() {
        sol_.t1_    = t_;
        sol_.y_     = y_;
        sol_.stats_ = st_;
        compute_sensitivities();
        return std::move(sol_);
    }

    [[nodiscard]] irk::expected<Solution<R>, Failure<R>> fail(Breakdown kind) {
        sol_.t1_    = t_;
        sol_.y_     = y_;
        sol_.stats_ = st_;
        compute_sensitivities();
        return irk::unexpected(Failure<R>{kind, t_, std::move(sol_)});
    }

    const F& f_;
    being::Interval<R> iv_;
    being::Measure<R> meas_;
    Options<R> opt_;
    std::size_t n_;
    R t_{};
    std::vector<R> y_, Z_;
    notion::StageSolver<R> solver_;
    std::vector<std::shared_ptr<const essence::ButcherTableau<R>>> tabs_;
    Solution<R> sol_;
    Stats st_;
};

} // namespace detail

template <being::Real R, typename F>
    requires notion::VectorField<F, R>
[[nodiscard]] irk::expected<Solution<R>, Failure<R>>
integrate(const F& field, being::Interval<R> iv, std::span<const R> y0,
          being::Measure<R> meas = {}, Options<R> opt = {}) {
    return detail::Integrator<R, F>(field, iv, y0, meas, std::move(opt)).run();
}

} // namespace irk::inference
