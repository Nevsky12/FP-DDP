#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · judgment.hpp                                                 [layer 4]
//
//  JUDGMENT (Urteil; Chelpanov: суждение).  A judgment predicates something
//  of a subject: "this step is acceptable", "this order is profitable",
//  "this magnitude befits a first step".  Each act here takes evidence
//  produced below (an error norm, a Newton report) and renders a verdict —
//  it never executes anything itself.
//
//  Exports:   StepControls, StepJudgment, judge_step,
//             OrderControls, OrderJudgment, initial_step.
// ─────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "being.hpp"
#include "notion/concepts.hpp"

namespace irk::judgment {

// ── Judgment of the step ────────────────────────────────────────────────────

template <being::Real R>
struct StepControls {
    R safety{R(0.9)};
    R min_factor{R(0.2)};
    R max_factor{R(5)};
};

template <being::Real R>
struct StepJudgment {
    bool accept{false};
    R    factor{1};              // multiplier for the next step magnitude
};

// The classical I-controller  h_new = h · safety · err^{-1/q}, clamped.
// After a rejection the factor may not exceed 1: a failed claim must not
// immediately be raised again.
template <being::Real R>
[[nodiscard]] StepJudgment<R> judge_step(R err, unsigned q, const StepControls<R>& ctl,
                                         bool rejected_before) noexcept {
    if (!std::isfinite(err)) return {.accept = false, .factor = R(0.25)};
    const R cap = rejected_before ? R(1) : ctl.max_factor;
    R f = ctl.safety * std::pow(err, R(-1) / R(q));   // err = 0 → ∞ → clamped
    f   = std::clamp(f, ctl.min_factor, cap);
    return {.accept = err <= R(1), .factor = f};
}

// ── Judgment of the order ───────────────────────────────────────────────────
//
// The rung ladder is the dialectic of the method itself: convergence
// behaviour of the Newton iteration (how *easily* the implicit stages are
// resolved) decides whether a richer or poorer formula is warranted.

struct OrderControls {
    std::size_t promote_after{3};   // consecutive easy steps before promotion
    std::size_t easy_iterations{2}; // ≤ this many Newton iterations is "easy"
    double      easy_theta{0.05};   // contraction below this is "easy"
    double      hard_theta{0.5};    // contraction above this demands demotion
};

class OrderJudgment {
public:
    explicit OrderJudgment(std::size_t rung_count, std::size_t start = 0)
        : top_(rung_count == 0 ? 0 : rung_count - 1), rung_(std::min(start, top_)) {}

    [[nodiscard]] std::size_t rung() const noexcept { return rung_; }

    void demote() noexcept {
        if (rung_ > 0) --rung_;
        streak_ = 0;
    }

    void on_accept(std::size_t iterations, double theta, const OrderControls& ctl,
                   std::size_t newton_max) noexcept {
        if (theta >= ctl.hard_theta || iterations + 1 >= newton_max) {
            demote();
            return;
        }
        const bool easy = iterations <= ctl.easy_iterations || theta <= ctl.easy_theta;
        streak_ = easy ? streak_ + 1 : 0;
        if (streak_ >= ctl.promote_after && rung_ < top_) {
            ++rung_;
            streak_ = 0;
        }
    }

private:
    std::size_t top_;
    std::size_t rung_;
    std::size_t streak_{0};
};

// ── Judgment of the beginning ───────────────────────────────────────────────
//
// Hairer–Wanner II, algorithm I.4: estimate ‖y″‖ by one Euler probe and choose
// h so that the local error of a method of consistency order q is ≈ 10⁻².

template <being::Real R, typename F>
    requires notion::VectorField<F, R>
[[nodiscard]] R initial_step(const F& f, R t0, std::span<const R> y0,
                             std::span<const R> f0, R dir, unsigned q,
                             const being::Measure<R>& meas, R hmax, R span_len,
                             std::size_t& nfev) {
    const R d0 = meas.norm(y0, y0);
    const R d1 = meas.norm(f0, y0);
    R h0 = (d0 < R(1e-5) || d1 < R(1e-5)) ? R(1e-6) : R(0.01) * d0 / d1;
    h0   = std::min(h0, span_len);

    std::vector<R> y1(y0.begin(), y0.end());
    being::axpy(dir * h0, f0, std::span<R>(y1));
    std::vector<R> f1(y0.size());
    f(t0 + dir * h0, std::span<const R>(y1), std::span<R>(f1));
    ++nfev;

    for (std::size_t i = 0; i < f1.size(); ++i) f1[i] -= f0[i];
    const R d2 = meas.norm(f1, y0) / h0;

    const R dm = std::max(d1, d2);
    const R h1 = (dm <= R(1e-15)) ? std::max(R(1e-6), h0 * R(1e-3))
                                  : std::pow(R(0.01) / dm, R(1) / R(q + 1));

    return std::min({R(100) * h0, h1, hmax, span_len});
}

} // namespace irk::judgment
