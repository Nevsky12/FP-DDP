#!/usr/bin/env python3
# acados side of the unicycle comparison (3 states, 2 controls), DDP solver.
# Same problem as test/cmp_unicycle.cpp. Writes logs/cmp_unicycle_acados.txt.
# See compare/README.md for how to run (needs t_renderer).
import os
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

N, dt = 40, 0.05
q = [1., 1., 0.5]
r = [0.1, 0.1]
qf = [50., 50., 20.]


def export_model():
    px = ca.SX.sym('px'); py = ca.SX.sym('py'); th = ca.SX.sym('th')
    x = ca.vertcat(px, py, th)
    v = ca.SX.sym('v'); w = ca.SX.sym('w'); u = ca.vertcat(v, w)
    xdot = ca.SX.sym('xdot', 3)
    f = ca.vertcat(v * ca.cos(th), v * ca.sin(th), w)
    mo = AcadosModel(); mo.name = 'unicycle'
    mo.x, mo.u, mo.xdot = x, u, xdot
    mo.f_expl_expr = f; mo.f_impl_expr = xdot - f
    return mo


def main():
    ocp = AcadosOcp(); ocp.model = export_model()
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    ocp.cost.cost_type = 'LINEAR_LS'; ocp.cost.cost_type_e = 'LINEAR_LS'
    Vx = np.zeros((5, 3)); Vx[:3, :3] = np.eye(3)
    Vu = np.zeros((5, 2)); Vu[3, 0] = 1.; Vu[4, 1] = 1.
    ocp.cost.Vx = Vx; ocp.cost.Vu = Vu
    ocp.cost.W = np.diag(q + r) * dt; ocp.cost.yref = np.zeros(5)
    ocp.cost.Vx_e = np.eye(3); ocp.cost.W_e = np.diag(qf); ocp.cost.yref_e = np.zeros(3)

    ocp.constraints.x0 = np.array([1.0, 0.5, 0.8])

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

    sol = AcadosOcpSolver(ocp, json_file='acados_unicycle.json')
    for i in range(N):
        sol.cost_set(i, "scaling", 1.0)
    st = sol.solve()
    X = np.array([sol.get(i, "x") for i in range(N + 1)])
    U = np.array([sol.get(i, "u") for i in range(N)])
    try:
        cost = float(sol.get_cost())
    except Exception:
        cost = float('nan')
    it = -1
    for k in ('nlp_iter', 'ddp_iter', 'sqp_iter'):
        try:
            it = int(sol.get_stats(k)); break
        except Exception:
            pass

    os.makedirs('logs', exist_ok=True)
    with open('logs/cmp_unicycle_acados.txt', 'w') as fp:
        fp.write(f"solver acados\nproblem unicycle\nN {N}\ndt {dt}\nstatus {st}\niters {it}\ncost {cost:.12e}\nstatio 0\ntraj\n")
        for i in range(N + 1):
            u0 = U[i, 0] if i < N else 0.0
            u1 = U[i, 1] if i < N else 0.0
            fp.write(f"{i} {X[i,0]:.10f} {X[i,1]:.10f} {X[i,2]:.10f} {u0:.10f} {u1:.10f}\n")
    print(f"acados unicycle: status={st} iters={it} cost={cost:.10e} xN={np.round(X[N],4)}")


if __name__ == '__main__':
    main()
