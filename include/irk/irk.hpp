#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  irk · irk.hpp                                                      [summit]
//
//  THE WHOLE.  "Das Wahre ist das Ganze."  The umbrella header gathers the
//  ascent — being → essence → notion → judgment → inference — and re-exports
//  its results under the single name irk, so a user of the system needs no
//  knowledge of the ladder that produced it (though it remains in plain view).
// ─────────────────────────────────────────────────────────────────────────────
#include "irk/being.hpp"
#include "irk/essence/algebra.hpp"
#include "irk/essence/collocation.hpp"
#include "irk/essence/orthogonal.hpp"
#include "irk/inference.hpp"
#include "irk/judgment.hpp"
#include "irk/notion/concepts.hpp"
#include "irk/notion/stages.hpp"

namespace irk {

// being
using being::Interval;
using being::Measure;
using being::Real;

// essence
using essence::ButcherTableau;
using essence::Family;
using essence::Matrix;
using essence::name;            // name(Family)

// notion
using notion::NewtonControls;
using notion::ProvidesJacobian;
using notion::VectorField;

// judgment
using judgment::OrderControls;
using judgment::StepControls;

// inference
using inference::Breakdown;
using inference::Failure;
using inference::integrate;
using inference::name;          // name(Breakdown)
using inference::Node;
using inference::Options;
using inference::Solution;
using inference::Stats;

} // namespace irk
