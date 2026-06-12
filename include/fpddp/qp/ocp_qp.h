#pragma once
// ─────────────────────────────────────────────────────────────────────────────
//  fpddp · qp/ocp_qp.h
//
//  Thin RAII C++20 wrapper over hpipm's OCP-QP IPM.  hpipm's OCP interface takes
//  plain column-major double* arrays for both data and solution (blasfeo lives
//  inside hpipm), so no blasfeo wrapper is needed at this boundary.
//
//  Models the stagewise QP
//      min  Σ_i ½[x_i;u_i]'[[Q_i,S_i'],[S_i,R_i]][x_i;u_i] + [q_i;r_i]'[x_i;u_i]
//      s.t. x_{i+1} = A_i x_i + B_i u_i + b_i,   x_0 fixed (set via setx0).
//  After solve(): get x/u/pi and the Riccati feedback K_i (u = K_i x + k_i) and
//  feedforward k_i — the data the DDP forward rollout consumes.
// ─────────────────────────────────────────────────────────────────────────────
#include <cstdlib>
#include <numeric>
#include <vector>

#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_d_ocp_qp_ipm.h>

namespace fpddp {

class OcpQp {
public:
    // nx, nu have length N+1 (nu[N] is typically 0). The initial state is fixed
    // by two-sided bounds on stage-0 state (set their value with setx0()).
    OcpQp(int N, std::vector<int> nx, std::vector<int> nu)
        : N_(N), nx_(std::move(nx)), nu_(std::move(nu)) {
        nbx_.assign(N_ + 1, 0); nbu_.assign(N_ + 1, 0);
        ng_.assign(N_ + 1, 0);  ns_.assign(N_ + 1, 0);
        nbx_[0] = nx_[0];                                   // fix x_0

        dim_mem_ = std::malloc(d_ocp_qp_dim_memsize(N_));
        d_ocp_qp_dim_create(N_, &dim_, dim_mem_);
        d_ocp_qp_dim_set_all(nx_.data(), nu_.data(), nbx_.data(), nbu_.data(),
                             ng_.data(), ns_.data(), &dim_);

        qp_mem_ = std::malloc(d_ocp_qp_memsize(&dim_));
        d_ocp_qp_create(&dim_, &qp_, qp_mem_);
        idxbx0_.resize(nx_[0]);
        std::iota(idxbx0_.begin(), idxbx0_.end(), 0);
        d_ocp_qp_set_idxbx(0, idxbx0_.data(), &qp_);

        sol_mem_ = std::malloc(d_ocp_qp_sol_memsize(&dim_));
        d_ocp_qp_sol_create(&dim_, &sol_, sol_mem_);

        arg_mem_ = std::malloc(d_ocp_qp_ipm_arg_memsize(&dim_));
        d_ocp_qp_ipm_arg_create(&dim_, &arg_, arg_mem_);
        // acados defaults: hpipm_mode BALANCE, qp_solver_iter_max 50, ric_alg 1
        d_ocp_qp_ipm_arg_set_default(BALANCE, &arg_);
        int qp_iter_max = 50; d_ocp_qp_ipm_arg_set_iter_max(&qp_iter_max, &arg_);
        int ric = 1; d_ocp_qp_ipm_arg_set_ric_alg(&ric, &arg_);  // sqrt Riccati → ric getters

        ws_mem_ = std::malloc(d_ocp_qp_ipm_ws_memsize(&dim_, &arg_));
        d_ocp_qp_ipm_ws_create(&dim_, &arg_, &ws_, ws_mem_);
    }

    ~OcpQp() {
        std::free(ws_mem_); std::free(arg_mem_); std::free(sol_mem_);
        std::free(qp_mem_); std::free(dim_mem_);
    }
    OcpQp(const OcpQp&) = delete;
    OcpQp& operator=(const OcpQp&) = delete;

    // stage data (column-major).  A_i: nx_{i+1}×nx_i, B_i: nx_{i+1}×nu_i,
    // Q_i: nx_i×nx_i, R_i: nu_i×nu_i, S_i: nu_i×nx_i.  b_i,q_i,r_i are vectors.
    void setA(int i, const double* m) { d_ocp_qp_set_A(i, const_cast<double*>(m), &qp_); }
    void setB(int i, const double* m) { d_ocp_qp_set_B(i, const_cast<double*>(m), &qp_); }
    void setb(int i, const double* v) { d_ocp_qp_set_b(i, const_cast<double*>(v), &qp_); }
    void setQ(int i, const double* m) { d_ocp_qp_set_Q(i, const_cast<double*>(m), &qp_); }
    void setR(int i, const double* m) { d_ocp_qp_set_R(i, const_cast<double*>(m), &qp_); }
    void setS(int i, const double* m) { d_ocp_qp_set_S(i, const_cast<double*>(m), &qp_); }
    void setq(int i, const double* v) { d_ocp_qp_set_q(i, const_cast<double*>(v), &qp_); }
    void setr(int i, const double* v) { d_ocp_qp_set_r(i, const_cast<double*>(v), &qp_); }
    void setx0(const double* x0) {
        d_ocp_qp_set_lbx(0, const_cast<double*>(x0), &qp_);
        d_ocp_qp_set_ubx(0, const_cast<double*>(x0), &qp_);
    }

    int  solve()  { d_ocp_qp_ipm_solve(&qp_, &sol_, &arg_, &ws_);
                    int s; d_ocp_qp_ipm_get_status(&ws_, &s); return s; }
    int  iter()   { int it; d_ocp_qp_ipm_get_iter(&ws_, &it); return it; }

    void getx(int i, double* v)  { d_ocp_qp_sol_get_x(i, &sol_, v); }
    void getu(int i, double* v)  { d_ocp_qp_sol_get_u(i, &sol_, v); }
    void getpi(int i, double* v) { d_ocp_qp_sol_get_pi(i, &sol_, v); }      // costate, length nx_{i+1}
    void getlam_lbx0(double* v)  { d_ocp_qp_sol_get_lam_lbx(0, &sol_, v); } // x0 lower-bound multiplier
    void getlam_ubx0(double* v)  { d_ocp_qp_sol_get_lam_ubx(0, &sol_, v); } // x0 upper-bound multiplier
    void getK(int i, double* v)  { d_ocp_qp_ipm_get_ric_K(&qp_, &arg_, &ws_, i, v); } // nu_i×nx_i col-major
    void getk(int i, double* v)  { d_ocp_qp_ipm_get_ric_k(&qp_, &arg_, &ws_, i, v); } // nu_i
    void getric_p(int i, double* v) { d_ocp_qp_ipm_get_ric_p(&qp_, &arg_, &ws_, i, v); } // nx_i (costate seed)

private:
    int N_;
    std::vector<int> nx_, nu_, nbx_, nbu_, ng_, ns_, idxbx0_;
    void *dim_mem_, *qp_mem_, *sol_mem_, *arg_mem_, *ws_mem_;
    struct d_ocp_qp_dim     dim_;
    struct d_ocp_qp         qp_;
    struct d_ocp_qp_sol     sol_;
    struct d_ocp_qp_ipm_arg arg_;
    struct d_ocp_qp_ipm_ws  ws_;
};

} // namespace fpddp
