#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  acados side of the fp-ddp stress comparison.  Problem definitions MUST match
#  test/stress_fpddp.cpp exactly (dynamics, weights, N, dt, x0, u_ref).
#
#  Settings (= the fp-ddp side):
#    DDP + MERIT_BACKTRACKING + adaptive Levenberg-Marquardt,
#    tol_stat = tol_eq = 1e-6, max_iter = 300,
#    IRK Gauss-Legendre s=3, 1 step, Newton tol 1e-12 (tight, to match the
#    vendored IRK's 1e-12 measure), cost scaling 1 (dt folded into W),
#    explicit acados-default init: x_i = x0, u = 0.
#
#  Run:
#    export ACADOS_SOURCE_DIR=$HOME/projects/acados
#    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
#    python3 compare/stress_acados.py <problem|all>
#  Writes logs/stress_<name>_acados.txt (header + per-iteration stats + traj).
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel


def model_pendulum():
    x = ca.SX.sym('x', 2); u = ca.SX.sym('u', 1)
    f = ca.vertcat(x[1], -ca.sin(x[0]) + u[0])
    return x, u, f

def model_dintegrator():
    x = ca.SX.sym('x', 2); u = ca.SX.sym('u', 1)
    f = ca.vertcat(x[1], u[0])
    return x, u, f

def model_vdp(mu=1.0):
    x = ca.SX.sym('x', 2); u = ca.SX.sym('u', 1)
    f = ca.vertcat(x[1], mu*(1 - x[0]**2)*x[1] - x[0] + u[0])
    return x, u, f

def model_unicycle():
    x = ca.SX.sym('x', 3); u = ca.SX.sym('u', 2)
    f = ca.vertcat(u[0]*ca.cos(x[2]), u[0]*ca.sin(x[2]), u[1])
    return x, u, f

def model_cartpole():
    M, m, g, l = 1.0, 0.1, 9.81, 0.8
    x = ca.SX.sym('x', 4); u = ca.SX.sym('u', 1)
    th, om, F = x[1], x[3], u[0]
    ct, st = ca.cos(th), ca.sin(th)
    den = M + m - m*ct*ct
    f = ca.vertcat(x[2], om,
                   (-m*l*st*om*om + m*g*ct*st + F)/den,
                   (-m*l*ct*st*om*om + F*ct + (M+m)*g*st)/(l*den))
    return x, u, f

def model_quadrotor():
    m, J, larm, g = 1.0, 0.05, 0.2, 9.81
    x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 2)
    th, F = x[2], u[0] + u[1]
    f = ca.vertcat(x[3], x[4], x[5],
                   -F*ca.sin(th)/m, F*ca.cos(th)/m - g, larm*(u[0]-u[1])/J)
    return x, u, f


UH = 1.0 * 9.81 / 2.0   # quadrotor hover thrust per rotor

PROBLEMS = {
    #  name:           (model,            N,  dt,   x0,                      q,                  ru,         qf,                       uref)
    'pendulum':      (model_pendulum,    40, 0.05, [2.0, 0.0],              [1, 0.1],           [0.01],     [50, 50],                 [0]),
    'dintegrator':   (model_dintegrator, 30, 0.05, [1.0, 0.0],              [1, 1],             [0.1],      [5, 5],                   [0]),
    'vdp':           (model_vdp,         50, 0.10, [2.0, 0.0],              [1, 1],             [0.1],      [10, 10],                 [0]),
    'unicycle':      (model_unicycle,    40, 0.05, [1.0, 0.5, 0.8],         [1, 1, 0.5],        [0.1, 0.1], [50, 50, 20],             [0, 0]),
    'cartpole':      (model_cartpole,    30, 0.04, [0.25, 0.08, 0.0, 0.0],  [10, 10, 0.1, 0.1], [0.01],     [100, 100, 10, 10],       [0]),
    'cartpole_long': (model_cartpole,    50, 0.04, [0.30, 0.15, 0.0, 0.0],  [10, 10, 0.1, 0.1], [0.01],     [100, 100, 10, 10],       [0]),
    'quadrotor':     (model_quadrotor,   40, 0.05, [0.5, 0.5, 0.3, 0, 0, 0],[10, 10, 10, 1, 1, 1],[0.1, 0.1],[100, 100, 100, 10, 10, 10],[UH, UH]),
    # hard cases meant to force backtracking (alpha < 1) and LM growth
    'pendulum_hard': (model_pendulum,    60, 0.05, [3.1, 0.0],              [1, 0.1],           [0.001],    [100, 100],               [0]),
    'vdp_hard':      (lambda: model_vdp(3.0), 60, 0.10, [3.0, 1.0],         [1, 1],             [0.01],     [10, 10],                 [0]),
    'cartpole_swing':(model_cartpole,    50, 0.04, [0.0, 2.5, 0.0, 0.0],    [1, 5, 0.1, 0.1],   [0.01],     [100, 100, 10, 10],       [0]),
}


def run(name):
    model_fn, N, dt, x0, q, ru, qf, uref = PROBLEMS[name]
    x, u, f = model_fn()
    nx, nu = x.shape[0], u.shape[0]

    m = AcadosModel()
    m.name = f'st_{name}'
    m.x, m.u = x, u
    xdot = ca.SX.sym('xdot', nx)
    m.xdot = xdot
    m.f_expl_expr = f
    m.f_impl_expr = xdot - f

    ocp = AcadosOcp()
    ocp.model = m
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt

    # LINEAR_LS == fp-ddp discrete cost (dt folded into W, scaling forced to 1)
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ny = nx + nu
    Vx = np.zeros((ny, nx)); Vx[:nx, :] = np.eye(nx)
    Vu = np.zeros((ny, nu)); Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx, ocp.cost.Vu = Vx, Vu
    ocp.cost.W = np.diag(list(q) + list(ru)) * dt
    ocp.cost.yref = np.concatenate([np.zeros(nx), np.array(uref, dtype=float)])
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.W_e = np.diag(qf)
    ocp.cost.yref_e = np.zeros(nx)

    ocp.constraints.x0 = np.array(x0, dtype=float)

    o = ocp.solver_options
    o.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    o.qp_solver_cond_N = N
    o.hessian_approx = 'GAUSS_NEWTON'
    o.integrator_type = 'IRK'
    o.sim_method_num_stages = 3
    o.sim_method_num_steps = 1
    o.collocation_type = 'GAUSS_LEGENDRE'
    o.sim_method_newton_iter = 20
    o.sim_method_newton_tol = 1e-12        # match the vendored IRK measure
    o.nlp_solver_type = 'DDP'
    o.nlp_solver_max_iter = 300
    o.nlp_solver_tol_stat = 1e-6
    o.nlp_solver_tol_eq = 1e-6
    o.globalization = 'MERIT_BACKTRACKING'
    o.with_adaptive_levenberg_marquardt = True

    code_dir = f'/tmp/stress_codegen_{name}'
    ocp.code_export_directory = code_dir
    sol = AcadosOcpSolver(ocp, json_file=f'{code_dir}/ocp.json')

    for i in range(N):
        sol.cost_set(i, "scaling", 1.0)
    # acados default init, set explicitly: x_i = x0, u = 0
    for i in range(N + 1):
        sol.set(i, "x", np.array(x0, dtype=float))
    for i in range(N):
        sol.set(i, "u", np.zeros(nu))

    status = sol.solve()
    iters = int(sol.get_stats('nlp_iter'))
    cost = float(sol.get_cost())
    res = sol.get_stats('residuals')       # [stat, eq, ineq, comp] at final iterate
    stats = sol.get_stats('statistics')    # (stat_n+1, iters+1): [it; res_stat; res_eq; cost; lm_prev; qp_status; qp_iter; ...]

    X = np.array([sol.get(i, "x") for i in range(N + 1)])
    U = np.array([sol.get(i, "u") for i in range(N)])

    os.makedirs('logs', exist_ok=True)
    path = f'logs/stress_{name}_acados.txt'
    with open(path, 'w') as fp:
        fp.write(f"solver acados\nproblem {name}\nN {N}\ndt {dt:.6f}\nstatus {status}\niters {iters}\n")
        fp.write(f"cost {cost:.14e}\nres_stat {res[0]:.6e}\nres_eq {res[1]:.6e}\n")
        fp.write("iterstats\n")            # it res_stat res_eq cost lm_prev qp_status qp_iter
        ncols = stats.shape[1]
        for i in range(ncols):
            fp.write(f"{int(stats[0, i])} {stats[1, i]:.10e} {stats[2, i]:.10e} {stats[3, i]:.14e} "
                     f"{stats[4, i]:.10e} {int(stats[5, i])} {int(stats[6, i])}\n")
        fp.write("traj\n")
        for i in range(N + 1):
            row = [f"{X[i, k]:.10f}" for k in range(nx)]
            row += [f"{U[i, j]:.10f}" if i < N else "0.0000000000" for j in range(nu)]
            fp.write(f"{i} " + " ".join(row) + "\n")

    print(f"acados {name:<14s} status={status} iters={iters:3d} cost={cost:.12e} "
          f"res_stat={res[0]:.2e} res_eq={res[1]:.2e}")
    # non-convergence is data for the comparison, not a harness failure
    # (cartpole_long is GN-limited; cartpole_swing is a knife-edge case, see stress_compare.py)
    return 0 if status == 0 or name in ('cartpole_long', 'cartpole_swing') else 1


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = list(PROBLEMS) if want == 'all' else [want]
    rc = 0
    for n in names:
        rc |= run(n)
    return rc


if __name__ == '__main__':
    sys.exit(main())
