#!/usr/bin/env python3
# acados side of the cart-pole comparison — acados' own pendulum_ode model
# (M=1, m=0.1, l=0.8, g=9.81; θ=0 upright/unstable), DDP solver.
# Same problem as test/cmp_cartpole.cpp. Writes logs/cmp_cartpole_acados.txt.
# See compare/README.md for how to run (needs t_renderer).
import os
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

N, dt = 30, 0.04
q = [10., 10., 0.1, 0.1]
r = 0.01
qf = [100., 100., 10., 10.]


def export_model():
    M, m, g, l = 1., 0.1, 9.81, 0.8
    x1 = ca.SX.sym('x1'); th = ca.SX.sym('theta'); v = ca.SX.sym('v'); dth = ca.SX.sym('dth')
    x = ca.vertcat(x1, th, v, dth)
    F = ca.SX.sym('F'); u = ca.vertcat(F)
    xdot = ca.SX.sym('xdot', 4)
    ct, st = ca.cos(th), ca.sin(th)
    den = M + m - m * ct * ct
    f = ca.vertcat(v, dth,
                   (-m * l * st * dth * dth + m * g * ct * st + F) / den,
                   (-m * l * ct * st * dth * dth + F * ct + (M + m) * g * st) / (l * den))
    mo = AcadosModel(); mo.name = 'cartpole'
    mo.x, mo.u, mo.xdot = x, u, xdot
    mo.f_expl_expr = f; mo.f_impl_expr = xdot - f
    return mo


def main():
    ocp = AcadosOcp(); ocp.model = export_model()
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    ocp.cost.cost_type = 'LINEAR_LS'; ocp.cost.cost_type_e = 'LINEAR_LS'
    Vx = np.zeros((5, 4)); Vx[:4, :4] = np.eye(4)
    Vu = np.zeros((5, 1)); Vu[4, 0] = 1.
    ocp.cost.Vx = Vx; ocp.cost.Vu = Vu
    ocp.cost.W = np.diag(q + [r]) * dt; ocp.cost.yref = np.zeros(5)
    ocp.cost.Vx_e = np.eye(4); ocp.cost.W_e = np.diag(qf); ocp.cost.yref_e = np.zeros(4)

    ocp.constraints.x0 = np.array([0.25, 0.08, 0., 0.])

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

    sol = AcadosOcpSolver(ocp, json_file='acados_cartpole.json')
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
    with open('logs/cmp_cartpole_acados.txt', 'w') as fp:
        fp.write(f"solver acados\nproblem cartpole\nN {N}\ndt {dt}\nstatus {st}\niters {it}\ncost {cost:.12e}\nstatio 0\ntraj\n")
        for i in range(N + 1):
            ui = U[i, 0] if i < N else 0.0
            fp.write(f"{i} {X[i,0]:.10f} {X[i,1]:.10f} {X[i,2]:.10f} {X[i,3]:.10f} {ui:.10f}\n")
    print(f"acados cartpole: status={st} iters={it} cost={cost:.10e} xN={np.round(X[N],4)}")


if __name__ == '__main__':
    main()
