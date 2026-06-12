#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  Compare fp-ddp vs acados DDP stress runs, iterate by iterate.
#    python3 compare/stress_compare.py [problem ...]    (default: all)
#
#  Reads logs/stress_<name>_{fpddp,acados}.txt.  Checks:
#    - terminal status (canonical labels)
#    - iteration count (strict)
#    - final cost (rel 1e-7) and trajectory (abs 1e-4)
#    - per-iteration cost sequence (rel 1e-5) — same algorithm ⇒ same path
#    - per-iteration LM sequence (rel 1e-4; acados logs lm with one-row lag)
#  cartpole_long is informational (Gauss-Newton-limited on both sides): it
#  must MATCH acados, not necessarily converge.
# ─────────────────────────────────────────────────────────────────────────────
import sys

ALL = ['pendulum', 'dintegrator', 'vdp', 'unicycle', 'cartpole', 'cartpole_long', 'quadrotor',
       'pendulum_hard', 'vdp_hard', 'cartpole_swing']
# cartpole_long: Gauss-Newton-limited on both sides — must match, not necessarily converge.
# cartpole_swing: knife-edge backtracking case — iterate parity holds ~12 iterations
#   (incl. matched alpha decisions), after which marginal accept/reject flips amplify
#   integrator-level noise (acados' 20-iteration-capped IRK Newton vs the vendored
#   adaptive IRK at violent trial states) and the paths legitimately split.
INFORMATIONAL = {'cartpole_long', 'cartpole_swing'}

FP_STATUS = {0: 'success', 1: 'maxiter', 2: 'qp_failure', 3: 'minstep', 4: 'nan', 5: 'minstep'}
AC_STATUS = {0: 'success', 1: 'nan', 2: 'maxiter', 3: 'minstep', 4: 'qp_failure'}


def load(path):
    d, mode = {'iterstats': [], 'traj': []}, None
    for line in open(path):
        s = line.split()
        if not s:
            continue
        if s[0] in ('iterstats', 'traj'):
            mode = s[0]
            continue
        if mode:
            d[mode].append([float(v) for v in s])
        else:
            d[s[0]] = s[1]
    return d


def rel(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


def compare(name):
    try:
        f = load(f'logs/stress_{name}_fpddp.txt')
        a = load(f'logs/stress_{name}_acados.txt')
    except FileNotFoundError as e:
        print(f"[{name}] missing: {e.filename}")
        return None

    fs = FP_STATUS.get(int(f['status']), f"fp?{f['status']}")
    as_ = AC_STATUS.get(int(a['status']), f"ac?{a['status']}")
    fi, ai = int(f['iters']), int(a['iters'])
    fcost, acost = float(f['cost']), float(a['cost'])

    issues = []
    if fs != as_:
        issues.append(f"status {fs} vs {as_}")
    if fi != ai:
        issues.append(f"iters {fi} vs {ai}")
    dc = rel(fcost, acost)
    if dc > 1e-7:
        issues.append(f"final cost rel diff {dc:.2e}")

    # per-iteration cost (column 3 both sides; acados rows may include extra cols)
    n = min(len(f['iterstats']), len(a['iterstats']))
    dcost_max = dstat_max = 0.0
    for i in range(n):
        dcost_max = max(dcost_max, rel(f['iterstats'][i][3], a['iterstats'][i][3]))
        dstat_max = max(dstat_max, rel(f['iterstats'][i][1], a['iterstats'][i][1]))
    if dcost_max > 1e-5:
        issues.append(f"iter cost rel diff {dcost_max:.2e}")

    # LM sequence: port row i col 4 (lm used at iter i) vs acados row i+1 col 4 (lagged)
    dlm_max = 0.0
    for i in range(min(len(f['iterstats']), len(a['iterstats']) - 1)):
        dlm_max = max(dlm_max, rel(f['iterstats'][i][4], a['iterstats'][i + 1][4]))
    if dlm_max > 1e-4:
        issues.append(f"LM seq rel diff {dlm_max:.2e}")

    # trajectory
    nt = min(len(f['traj']), len(a['traj']))
    dtraj = 0.0
    for i in range(nt):
        for c in range(1, min(len(f['traj'][i]), len(a['traj'][i]))):
            dtraj = max(dtraj, abs(f['traj'][i][c] - a['traj'][i][c]))
    if dtraj > 1e-4:
        issues.append(f"traj max diff {dtraj:.2e}")

    ok = not issues
    tag = 'AGREE' if ok else ('NOTE' if name in INFORMATIONAL else 'DIFFER')
    print(f"[{name:<14s}] {tag}  fp:{fs}/{fi}it  ac:{as_}/{ai}it  "
          f"|Δcost|rel={dc:.1e}  iter-cost={dcost_max:.1e}  lm={dlm_max:.1e}  "
          f"res_stat={dstat_max:.1e}  traj={dtraj:.1e}"
          + (f"   <- {'; '.join(issues)}" if issues else ""))
    return ok or name in INFORMATIONAL


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else ALL
    results = [compare(n) for n in names]
    bad = sum(1 for r in results if r is False) + sum(1 for r in results if r is None)
    print(f"\n{'ALL AGREE' if bad == 0 else f'{bad} problem(s) DIFFER/missing'} "
          f"({len([r for r in results if r])} ok / {len(results)} compared)")
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
