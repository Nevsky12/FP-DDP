#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  acados side of the fp-ddp comparison: the SAME unconstrained 2-state pendulum
#  regulation, solved with acados' DDP solver, written to logs/cmp_acados.txt.
#
#  Shared problem (identical to test/cmp_pendulum.cpp):
#    ẋ1 = x2,  ẋ2 = −sin(x1) + u            (no control bounds)
#    x0 = [2.0, 0.0],  N = 40, dt = 0.05
#    stage  ½(x1² + 0.1 x2² + 0.01 u²)·dt ,  terminal  ½·50·(x1² + x2²)
#    DDP / Gauss-Newton / IRK Gauss-Legendre s=3, 1 step/interval.
#  The LINEAR_LS cost with dt folded into W and scaling=1 reproduces fp-ddp's
#  discrete objective exactly; same integrator → the discrete OCPs are identical,
#  so both DDP solvers should reach the same optimum.
#
#  Run (needs acados libs on the path + the t_renderer codegen binary):
#    scripts/build_acados.sh                                   # libacados (done)
#    export ACADOS_SOURCE_DIR=$HOME/projects/acados
#    python -c "from acados_template.utils import get_tera; get_tera()"   # t_renderer
#    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
#    $HOME/projects/RocketInitialGuess/.venv/bin/python compare/acados_pendulum.py
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

N, dt = 40, 0.05
qx, qv, ru, qf = 1.0, 0.1, 0.01, 50.0


def export_model():
    x = ca.SX.sym('x', 2)
    u = ca.SX.sym('u', 1)
    xdot = ca.SX.sym('xdot', 2)
    f = ca.vertcat(x[1], -ca.sin(x[0]) + u[0])
    m = AcadosModel()
    m.name = 'pend2'
    m.x, m.u, m.xdot = x, u, xdot
    m.f_expl_expr = f
    m.f_impl_expr = xdot - f
    return m


def main():
    ocp = AcadosOcp()
    ocp.model = export_model()
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    # LINEAR_LS cost == fp-ddp discrete cost (dt folded into W, scaling = 1)
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.Vx = np.array([[1., 0.], [0., 1.], [0., 0.]])
    ocp.cost.Vu = np.array([[0.], [0.], [1.]])
    ocp.cost.W = np.diag([qx, qv, ru]) * dt
    ocp.cost.yref = np.zeros(3)
    ocp.cost.Vx_e = np.eye(2)
    ocp.cost.W_e = np.diag([qf, qf])
    ocp.cost.yref_e = np.zeros(2)

    ocp.constraints.x0 = np.array([2.0, 0.0])

    o = ocp.solver_options
    o.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    o.hessian_approx = 'GAUSS_NEWTON'
    o.integrator_type = 'IRK'
    o.sim_method_num_stages = 3
    o.sim_method_num_steps = 1
    o.collocation_type = 'GAUSS_LEGENDRE'
    o.nlp_solver_type = 'DDP'
    o.nlp_solver_max_iter = 300
    o.globalization = 'MERIT_BACKTRACKING'
    o.with_adaptive_levenberg_marquardt = True

    sol = AcadosOcpSolver(ocp, json_file='acados_pend2.json')
    for i in range(N):
        sol.cost_set(i, "scaling", 1.0)

    st = sol.solve()
    X = np.array([sol.get(i, "x") for i in range(N + 1)])
    U = np.array([sol.get(i, "u") for i in range(N)])
    try:
        cost = float(sol.get_cost())
    except Exception:
        cost = float('nan')
    iters = -1
    for key in ('nlp_iter', 'ddp_iter', 'sqp_iter'):
        try:
            iters = int(sol.get_stats(key))
            break
        except Exception:
            pass

    os.makedirs('logs', exist_ok=True)
    with open('logs/cmp_acados.txt', 'w') as fp:
        fp.write(f"solver acados\nN {N}\ndt {dt}\nstatus {st}\niters {iters}\ncost {cost:.12e}\nstatio 0\ntraj\n")
        for i in range(N + 1):
            ui = U[i, 0] if i < N else 0.0
            fp.write(f"{i} {X[i,0]:.10f} {X[i,1]:.10f} {ui:.10f}\n")
    print(f"acados: status={st} iters={iters} cost={cost:.10e} xN=[{X[N,0]:.6f} {X[N,1]:.6f}]")


if __name__ == '__main__':
    main()
