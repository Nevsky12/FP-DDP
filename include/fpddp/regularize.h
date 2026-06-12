#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · regularize.h
//
//  Stage-Hessian projection for exact-Hessian DDP (acados PROJECT analog):
//  symmetric eigendecomposition (cyclic Jacobi — stage blocks are ≤ ~10×10)
//  with eigenvalues clamped to ≥ min_eig, so the QP Hessian handed to hpipm's
//  Riccati pass is uniformly positive definite.
// ─────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstddef>
#include <vector>

#include "fpddp/model.h"

namespace fpddp {

// Eigendecompose the symmetric matrix W (overwritten) via cyclic Jacobi;
// returns eigenvalues in d, eigenvectors in the columns of V.
inline void jacobi_eig(Mat& W, std::vector<double>& d, Mat& V) {
    const std::size_t n = W.rows();
    V.set_zero();
    for (std::size_t i = 0; i < n; ++i) V(i, i) = 1.0;
    for (int sweep = 0; sweep < 50; ++sweep) {
        double off = 0.0;
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = p + 1; q < n; ++q) off += W(p, q) * W(p, q);
        if (off < 1e-26) break;
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = W(p, q);
                if (std::abs(apq) < 1e-300) continue;
                const double theta = (W(q, q) - W(p, p)) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) /
                                 (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (std::size_t k = 0; k < n; ++k) {        // rotate rows/cols p,q
                    const double wkp = W(k, p), wkq = W(k, q);
                    W(k, p) = c * wkp - s * wkq;
                    W(k, q) = s * wkp + c * wkq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double wpk = W(p, k), wqk = W(q, k);
                    W(p, k) = c * wpk - s * wqk;
                    W(q, k) = s * wpk + c * wqk;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double vkp = V(k, p), vkq = V(k, q);
                    V(k, p) = c * vkp - s * vkq;
                    V(k, q) = s * vkp + c * vkq;
                }
            }
    }
    d.resize(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = W(i, i);
}

// W ← V·diag(clamp(d))·Vᵀ for symmetric W (symmetrized on entry).
// mirror = false: λ ← max(λ, min_eig)   (acados PROJECT)
// mirror = true:  λ ← max(|λ|, min_eig) (acados MIRROR — keeps curvature scale)
inline void project_psd(Mat& W, double min_eig, bool mirror = false) {
    const std::size_t n = W.rows();
    for (std::size_t i = 0; i < n; ++i)                    // symmetrize
        for (std::size_t j = i + 1; j < n; ++j) {
            const double v = 0.5 * (W(i, j) + W(j, i));
            W(i, j) = W(j, i) = v;
        }
    Mat A = W, V(n, n);
    std::vector<double> d;
    jacobi_eig(A, d, V);
    bool clamped = false;
    for (auto& e : d) {
        const double v = mirror ? std::max(std::abs(e), min_eig) : std::max(e, min_eig);
        if (v != e) { e = v; clamped = true; }
    }
    if (!clamped) return;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0;
            for (std::size_t k = 0; k < n; ++k) acc += V(i, k) * d[k] * V(j, k);
            W(i, j) = acc;
        }
}

}  // namespace fpddp
