# irk — implicit Runge–Kutta integration as a ladder of notions

A header-only C++23 library implementing **variable-order implicit Runge–Kutta
methods** of the two classical collocation families — **Gauss–Legendre**
(order 2s, A-stable, symplectic) and **Radau IIA** (order 2s−1, L-stable,
stiffly accurate) — at **any stage count s ∈ [1, 16]**, with adaptive step
size, adaptive order, simplified-Newton stage resolution, a stiff-filtered
embedded error estimate, collocation dense output, and an `std::expected`
error channel that returns the partial trajectory on breakdown.

The request that shaped it: *build the system the way Hegel's «Wissenschaft
der Logik» and Chelpanov's «Учебник логики» build a science — from fundamental
instances upward, so the graph of terms is clean.* Here that is not
decoration; **the include graph is the table of contents of the argument.**

---

## 1 · The architecture is the argument

Each header is one moment of the dialectic. A header may include only what
stands strictly below it; every name is introduced exactly once, at the layer
where its concept first becomes thinkable.

```
                               irk.hpp                       the whole (das Ganze)
                                  │
                            inference.hpp                    INFERENCE · Schluss · умозаключение
                                  │                          the integrator: trajectory as conclusion
              ┌───────────────────┼───────────────────┐
        judgment.hpp        notion/stages.hpp          │     JUDGMENT · Urteil · суждение
        accept/reject,      stage equations +          │     NOTION (in action): Newton as syllogism
        order rung,         simplified Newton          │
        first step                │                    │
              │           notion/concepts.hpp          │     NOTION · Begriff · понятие
              └──────────► VectorField, Jacobian       │     definition per genus et differentiam
                                  │                    │
                                  │        essence/collocation.hpp    ESSENCE III: actuality
                                  │        integration_weights,       the tableau, generated
                                  │        ButcherTableau, Family     Family = division of the genus
                                  │         ┌─────────┴─────────┐
                                  │  essence/algebra.hpp  essence/orthogonal.hpp
                                  │  Matrix, PLU          Legendre, nodes
                                  │  ESSENCE I: ground    ESSENCE II: appearance
                                  └─────────┬─────────────────┘
                                        being.hpp             BEING · Sein
                                        Real, axpy,           quality, quantity,
                                        Interval, Measure     and Measure (Maß)
```

| layer | header | Hegel | Chelpanov | mathematics |
|---|---|---|---|---|
| 0 | `being.hpp` | Sein — quality, quantity, **Maß** | the material of all judgments | `Real`, `axpy`, `Interval`, `Measure` (scaled RMS norms) |
| 1 | `essence/algebra.hpp` | Wesen — ground & relation | — | `Matrix` (C++23 `[i, j]`), `PLU` via `std::expected` |
| 1 | `essence/orthogonal.hpp` | Wesen — appearance | — | `constexpr` Legendre recurrences (+ `static_assert` proofs), Gauss & Radau nodes by bracketed bisection |
| 1 | `essence/collocation.hpp` | Wesen — actuality | **division** of the concept (`Family`) | `integration_weights` (Björck–Pereyra), `ButcherTableau` |
| 2 | `notion/concepts.hpp` | Begriff — the universal | **definition** by genus & differentia, as literal C++ `concept`s | `VectorField`, `ProvidesJacobian`, divided-difference Jacobian |
| 2 | `notion/stages.hpp` | Begriff — judgment & syllogism in nuce | inference: law (Jacobian) + fact (residual) → correction | `StageSolver`: M = I − h(A⊗J), simplified Newton |
| 3 | `judgment.hpp` | **Urteil** | суждение — a verdict, never an act | `judge_step`, `OrderJudgment`, `initial_step` (HW I.4) |
| 4 | `inference.hpp` | **Schluss** | умозаключение — universal ∧ particular ⇒ singular | `integrate`: tableau + field + state → `Solution` |
| ⊤ | `irk.hpp` | das Ganze; *Aufhebung* | — | flat aliases; every layer preserved, none escapes |

The *Aufhebung* is literal: each layer is **negated** as self-sufficient (a
tableau computes nothing), **preserved** intact (you can use `ButcherTableau`
alone), and **elevated** into the next (the integrator quotes it verbatim).

## 2 · One generative principle

The entire method content of the library unfolds from a single operation
(`essence/collocation.hpp`):

```
integration_weights(c, θ):   find w with  Σⱼ wⱼ cⱼ^(k−1) = θᵏ/k,   k = 1..s
```

— i.e. the weights of the interpolatory quadrature ∫₀^θ on the nodes `c`,
solved as a dual Vandermonde system by the **Björck–Pereyra** algorithm
(O(s²), forward-stable on these increasing positive nodes — measured residuals
stay at machine epsilon up to s = 16). Varying **θ alone** yields:

| θ | result |
|---|---|
| θ = cᵢ | row i of the collocation matrix **A** |
| θ = 1 | the weights **b** |
| θ = 1 on c₁..c₋₁ | the embedded **b̂**, hence the defect **d = b − b̂** |
| θ = σ, any σ | **dense-output** weights w(σ) = A⁻ᵀ·∫₀^σ — which also serve as the Newton **predictor** (extrapolation of the previous step's collocation polynomial) |

The nodes themselves come from one Legendre recurrence: Gauss points are the
roots of P_s; Radau IIA points are the roots of (P_s − P_{s−1})/(x − 1) plus
the endpoint. Two species, one genus — Chelpanov's *деление понятия* as an
`enum class Family`.

## 3 · The mathematics, briefly

**Stages.** Each step solves for the stage displacements Z (stage-major)

```
Zᵢ = h Σⱼ aᵢⱼ f(t + cⱼh, y + Zⱼ),         y₁ = y + Σ b̃ᵢ Zᵢ,   b̃ = A⁻ᵀ b
```

(for Radau IIA, b̃ = e_s exactly — stiff accuracy — and it is snapped so).
The simplified Newton iteration factors **M = I − h(A ⊗ J)** once per attempt;
the Jacobian (analytic if the field `ProvidesJacobian`, divided differences
otherwise) is refreshed once per step *point* and reused across retries.
Convergence is judged by the contraction θ = ‖Δₖ‖/‖Δₖ₋₁‖ through
η·‖Δ‖ ≤ κ with η = θ/(1−θ), with divergence and hopeless-contraction exits
(Hairer–Wanner IV.8).

**Error.** The embedded estimate is free:
`est = Σ d̃ᵢ Zᵢ ≡ h Σ dᵢ f(Yᵢ)` with d̃ = A⁻ᵀd, an order-s estimator. For
Radau IIA it is filtered through (I − hJ)⁻¹ — a deliberate, documented
simplification of RADAU5's γ₀-tuned filter — so that stiff components cannot
inflate it (sections 3–4 of the demo measure the consequence). The step
controller is the classical `h ← h·0.9·err^(−1/s)`, clamped to [0.2, 5],
growth capped at 1 immediately after a rejection.

**Order.** A ladder of tableaus (default stages {2, 3, 5, 7}: Radau orders
{3, 5, 9, 13}, Gauss {4, 6, 10, 14}) is climbed and descended on the evidence
of the Newton iteration itself: three consecutive *easy* solves (≤ 2
iterations or θ ≤ 0.05) promote; a hard solve (θ ≥ 0.5, iteration limit
grazed) or two consecutive Newton failures demote. The dialectic of the
method: how easily the implicit is made explicit decides how rich a formula
the moment can bear.

**Beginnings, ends, records.** The first step follows Hairer–Wanner
algorithm I.4. Integration runs in either direction. Every accepted step
stores its collocation polynomial, so `Solution::eval(t)` gives dense output —
superconvergent at the nodes, stage-order accurate between them. Breakdown
(step underflow, attempt budget, Newton failure at fixed step) returns
`Failure{kind, t, partial}` through `std::expected`; the partial trajectory
remains fully usable.

## 4 · Quickstart

```cpp
#include "irk/irk.hpp"

struct VanDerPol {
    double mu;
    void operator()(double, std::span<const double> y, std::span<double> dy) const {
        dy[0] = y[1];
        dy[1] = mu * (1 - y[0] * y[0]) * y[1] - y[0];
    }
    void jacobian(double, std::span<const double> y, irk::Matrix<double>& J) const {
        J[0, 1] = 1;                       // optional: omit the whole method and
        J[1, 0] = -2 * mu * y[0] * y[1] - 1;   // divided differences take over
        J[1, 1] = mu * (1 - y[0] * y[0]);
    }
};

int main() {
    std::vector<double> y0{2.0, 0.0};
    auto sol = irk::integrate(VanDerPol{1000.0},
                              irk::Interval<double>{0.0, 3000.0},
                              std::span<const double>(y0),
                              irk::Measure<double>{.rtol = 1e-8, .atol = 1e-8});
    if (!sol) return 1;                    // sol.error(): kind, t, partial trajectory
    auto y_mid = sol->eval(1500.0);        // dense output anywhere in the span
    auto& st   = sol->stats();             // nfev, njac, nlu, accepted_by_stage, …
}
```

Options worth knowing (`irk::Options<R>`): `family`, `stage_ladder`,
`first_step`, `max_step`, `fixed_step` (disables adaptivity), `fixed_stages`
(pins the order), plus `StepControls`, `OrderControls`, `NewtonControls`.

**Build** (GCC 13+, header-only):

```
g++ -std=c++23 -O2 -Wall -Wextra -Iinclude examples/demo.cpp -o demo
```

## 5 · What the demo measures (numbers from this machine)

The suite (`examples/demo.cpp`, ~30 ms, 15 PASS verdicts) is the empirical
counterpart of the construction:

**1 · Genesis of the tableau.** Generated Radau IIA s=3 matches the RADAU5
constants and Gauss s=2 matches Hammer–Hollingsworth to **1.1·10⁻¹⁶**;
order-condition residuals (B(s), C(s) moment identities) stay ≤ 2.2·10⁻¹⁶
for s = 2..7 — and remain at machine epsilon through s = 16.

**2 · Orders made visible.** Fixed-step slopes on the logistic equation:

```
family      s  order   err(h)       err(h/2)     slope   expected
Gauss       2     4    1.351e-07    8.448e-09     4.00   4
Gauss       3     6    1.745e-10    2.892e-12     5.92   6
Gauss       4     8    3.740e-12    3.658e-14     6.68   8   (rounding floor)
Radau IIA   2     3    1.305e-06    1.636e-07     3.00   3
Radau IIA   3     5    4.667e-09    1.426e-10     5.03   5
Radau IIA   4     7    3.300e-11    2.070e-13     7.32   7   (rounding floor)
```

**3 · A stiff scalar.** Prothero–Robinson with λ = −10⁶ on [0, 10]:
**11 accepted steps**, the largest of size 3.1 (≈ 3·10⁶ stiff time scales),
endpoint error 5.0·10⁻¹¹ at rtol = 10⁻⁸. Dense output: 1.2·10⁻⁹ at the
nodes (superconvergent), 1.9·10⁻⁴ between them (stage order, with h ≈ 3).

**4 · Van der Pol, μ = 1000, [0, 3000].** 336 accepted steps (140 error +
45 Newton rejections), 17 130 f-evaluations, 336 Jacobians. The order ladder
at work — accepted steps by stage count: s=2: 3, s=3: 3, s=5: 54,
**s=7 (order 13): 276**. Endpoint agrees with an rtol = 10⁻¹¹ reference to
2.8·10⁻⁸.

**5 · Kepler, e = 0.6, 50 periods, fixed h = 0.05, s = 3.** Energy drift
max|H − H₀|: **Gauss 1.8·10⁻⁸ (bounded — symplectic)** vs. Radau IIA
9.9·10⁻⁵ (secular) — a factor of **5632**.

## 6 · Choices and limits, honestly

- The (sn)×(sn) Newton matrix is factored **densely**: O(s³n³) per attempt.
  Production Radau codes diagonalize A into s complex n×n solves; here
  transparency of the Kronecker structure was chosen over that constant
  factor. Fine for small and medium n.
- The Jacobian is refreshed at every accepted step point (and reused across
  rejections at that point); there is no cross-step Jacobian-recycling
  heuristic.
- The error estimate has order s while the endpoint has order 2s−1 / 2s, so
  steps are chosen conservatively — the same species of compromise RADAU5
  makes with its order-3 estimate inside an order-5 method. The Radau filter
  (I − hJ)⁻¹ simplifies RADAU5's tuned γ₀ version.
- No event location, no mass matrices, no sparse Jacobians.
- `max_steps` budgets *attempts*, so rejection storms terminate; allocation
  inside the step loop (the per-attempt factorization) favours clarity over
  micro-optimization.
- Everything is templated on `Real`: instantiating with `long double` takes
  the tableau residuals from ~10⁻¹⁶ to ~5·10⁻²⁰ (measured), at no change to
  user code.

## 7 · References

- E. Hairer, G. Wanner, *Solving Ordinary Differential Equations II — Stiff
  and Differential-Algebraic Problems* (Radau IIA, simplified Newton and its
  convergence control, the (I − hγJ) filter, initial-step algorithm I.4).
- J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*
  (collocation, order theory, Gauss & Radau families).
- Å. Björck, V. Pereyra, *Solution of Vandermonde Systems of Equations*,
  Math. Comp. 24 (1970) — the weight solver.
- Г. И. Челпанов, *Учебник логики* — понятие, суждение, умозаключение;
  определение и деление понятий.
- G. W. F. Hegel, *Wissenschaft der Logik* — Sein, Wesen, Begriff; the ladder
  this library climbs.
