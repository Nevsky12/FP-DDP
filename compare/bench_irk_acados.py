#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  acados side of the IRK bench + the comparison report.  Case grids MUST match
#  test/bench_irk.cpp (joined on (model, case index)).
#
#  Per case: one GL3 (s=3) IRK step of length dt with forward sensitivities,
#  newton_iter = 20, newton_tol = 1e-12 (same as the stress setup).  acados'
#  IRK has no failure status — it silently returns the unconverged solution.
#  Reference: CVODES (casadi) at abstol = reltol = 1e-13.
#
#  Report per (model, dt):
#    - median step time (with sensitivities): fp-ddp vs acados
#    - port-vs-acados discrete-map agreement (both claim the same GL3 map)
#    - error vs CVODES (= discretization error when Newton converged)
#    - failures: fp-ddp flagged (detected) vs acados silent (err ≫ port err)
#
#  Run (after ./build/bench_irk):
#    export ACADOS_SOURCE_DIR=$HOME/projects/acados
#    export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
#    python3 compare/bench_irk_acados.py
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import casadi as ca
from acados_template import AcadosSim, AcadosSimSolver, AcadosModel


def model_pendulum():
    x = ca.SX.sym('x', 2); u = ca.SX.sym('u', 1)
    return x, u, ca.vertcat(x[1], -ca.sin(x[0]) + u[0])

def model_cartpole():
    M, m, g, l = 1.0, 0.1, 9.81, 0.8
    x = ca.SX.sym('x', 4); u = ca.SX.sym('u', 1)
    th, om, F = x[1], x[3], u[0]
    ct, st = ca.cos(th), ca.sin(th)
    den = M + m - m*ct*ct
    f = ca.vertcat(x[2], om, (-m*l*st*om*om + m*g*ct*st + F)/den,
                   (-m*l*ct*st*om*om + F*ct + (M+m)*g*st)/(l*den))
    return x, u, f

def model_vdp(mu=5.0):
    x = ca.SX.sym('x', 2); u = ca.SX.sym('u', 1)
    return x, u, ca.vertcat(x[1], mu*(1 - x[0]**2)*x[1] - x[0] + u[0])

MODELS = {'pendulum': model_pendulum, 'cartpole': model_cartpole, 'vdp': model_vdp}


def cases():
    cs = []
    k = 0
    for th in (0.5, 1.5, 3.0):
        for om in (0.0, 2.0, 8.0, 20.0):
            for uu in (0.0, 2.0):
                for dt in (0.05, 0.2, 0.5):
                    cs.append(('pendulum', k, dt, [th, om], [uu])); k += 1
    k = 0
    for th in (0.1, 1.5, 3.0):
        for om in (0.0, 5.0, 15.0, 40.0):
            for v in (0.0, 5.0):
                for F in (0.0, 20.0):
                    for dt in (0.04, 0.2):
                        cs.append(('cartpole', k, dt, [0.0, th, v, om], [F])); k += 1
    k = 0
    for x1 in (0.5, 2.0, 4.0):
        for x2 in (0.0, 3.0, 8.0):
            for dt in (0.1, 0.5):
                cs.append(('vdp', k, dt, [x1, x2], [0.0])); k += 1
    return cs


def make_sim_solver(name, dt):
    x, u, f = MODELS[name]()
    m = AcadosModel()
    m.name = f'bi_{name}_{str(dt).replace(".", "p")}'
    m.x, m.u = x, u
    xdot = ca.SX.sym('xdot', x.shape[0])
    m.xdot = xdot
    m.f_expl_expr = f
    m.f_impl_expr = xdot - f
    sim = AcadosSim()
    sim.model = m
    o = sim.solver_options
    o.integrator_type = 'IRK'
    o.num_stages = 3
    o.num_steps = 1
    o.collocation_type = 'GAUSS_LEGENDRE'
    o.newton_iter = 20
    o.newton_tol = 1e-12
    o.sens_forw = True
    o.T = dt
    code_dir = f'/tmp/bench_irk_{m.name}'
    sim.code_export_directory = code_dir
    return AcadosSimSolver(sim, json_file=f'{code_dir}/sim.json', verbose=False)


def make_reference(name, dt):
    x, u, f = MODELS[name]()
    dae = {'x': x, 'p': u, 'ode': f}
    return ca.integrator('R', 'cvodes', dae, 0.0, dt, {'abstol': 1e-13, 'reltol': 1e-13})


def load_port(path):
    out = {}
    for line in open(path):
        s = line.split()
        if not s or s[0] != 'case':
            continue
        model, idx, dt = s[1], int(s[2]), float(s[3])
        ok, leaves, t_cold, t_warm = int(s[4]), int(s[5]), float(s[6]), float(s[7])
        out[(model, idx)] = {'dt': dt, 'ok': ok, 'leaves': leaves, 't': t_cold, 't_warm': t_warm,
                             'x': np.array([float(v) for v in s[8:]])}
    return out


def main():
    port = load_port('logs/bench_irk_fpddp.txt')
    solvers, refs = {}, {}
    rows = []
    reps = 100
    for model, idx, dt, x0, u0 in cases():
        key = (model, dt)
        if key not in solvers:
            solvers[key] = make_sim_solver(model, dt)
            refs[key] = make_reference(model, dt)
        sol = solvers[key]
        sol.set('x', np.array(x0)); sol.set('u', np.array(u0))
        ts = []
        for _ in range(3):
            sol.solve()
        for _ in range(reps):
            status = sol.solve()
            ts.append(float(sol.get('time_tot')))
        xa = sol.get('x').copy()
        t_ac = float(np.median(ts)) * 1e9
        xr = np.array(refs[key](x0=np.array(x0), p=np.array(u0))['xf']).flatten()
        p = port[(model, idx)]
        scale = max(1.0, float(np.max(np.abs(xr))))
        rows.append({
            'model': model, 'idx': idx, 'dt': dt, 'status_ac': status,
            'ok_fp': p['ok'], 'leaves': p['leaves'], 't_fp': p['t'], 't_fp_warm': p['t_warm'],
            't_ac': t_ac,
            'd_fp_ac': float(np.max(np.abs(p['x'] - xa))) / scale if p['ok'] else np.nan,
            'e_fp': float(np.max(np.abs(p['x'] - xr))) / scale if p['ok'] else np.nan,
            'e_ac': float(np.max(np.abs(xa - xr))) / scale,
        })

    with open('logs/bench_irk_table.txt', 'w') as fp:
        hdr = (f"{'model':<10s} {'dt':>5s} {'n':>3s} | {'cold µs':>8s} {'warm µs':>8s} {'t_ac µs':>8s} | "
               f"{'agree(1-leaf)':>14s} | {'err_fp':>9s} {'err_ac':>9s} | "
               f"{'fpFAIL':>6s} {'fpSPLIT':>7s} {'acBAD':>6s}")
        print(hdr); fp.write(hdr + '\n')
        for model in MODELS:
            for dt in sorted({r['dt'] for r in rows if r['model'] == model}):
                g = [r for r in rows if r['model'] == model and r['dt'] == dt]
                okg = [r for r in g if r['ok_fp']]
                one = [r for r in okg if r['leaves'] == 1]          # same discrete map as acados
                both = [r for r in one if r['d_fp_ac'] < 1e-8]      # agree to the same root
                fp_fail = sum(1 for r in g if not r['ok_fp'])
                fp_split = sum(1 for r in okg if r['leaves'] > 1)
                # acados bad: NaN with status SUCCESS, or ≥10x farther from CVODES than the
                # port's verified-converged answer / flagged loudly by the port
                def ac_is_bad(r):
                    if not np.isfinite(r['e_ac']):
                        return True                       # NaN-as-SUCCESS
                    if r['ok_fp'] and np.isfinite(r['e_fp']):
                        return r['e_ac'] > 1e-3 and r['e_ac'] > 10 * r['e_fp']
                    return r['e_ac'] > 1e-3               # port flagged; acados silently off
                ac_bad = sum(1 for r in g if ac_is_bad(r))
                tf = np.median([r['t_fp'] for r in both]) / 1e3 if both else np.nan
                tw = np.median([r['t_fp_warm'] for r in both]) / 1e3 if both else np.nan
                ta = np.median([r['t_ac'] for r in both]) / 1e3 if both else np.nan
                ef = max((r['e_fp'] for r in both), default=np.nan)
                ea = max((r['e_ac'] for r in both), default=np.nan)
                line = (f"{model:<10s} {dt:>5.2f} {len(g):>3d} | {tf:>8.2f} {tw:>8.2f} {ta:>8.2f} | "
                        f"{len(both):>5d}/{len(one):<3d} agree | {ef:>9.1e} {ea:>9.1e} | "
                        f"{fp_fail:>6d} {fp_split:>7d} {ac_bad:>6d}")
                print(line); fp.write(line + '\n')
        # detail rows: anything not a clean 1-leaf mutual agreement (NaN-aware)
        det = [r for r in rows if (not r['ok_fp']) or r['leaves'] > 1 or not (r['d_fp_ac'] < 1e-8)]
        if det:
            print('\nnon-agreeing cases (err = max-abs vs CVODES, scaled):')
            fp.write('\nnon-agreeing cases:\n')
            for r in det:
                line = (f"  {r['model']}[{r['idx']:3d}] dt={r['dt']:.2f} fp_ok={r['ok_fp']} "
                        f"leaves={r['leaves']} fp-vs-ac={r['d_fp_ac']:.1e} "
                        f"err_fp={r['e_fp']:.1e} err_ac={r['e_ac']:.1e}")
                print(line); fp.write(line + '\n')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
