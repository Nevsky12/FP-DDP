#!/usr/bin/env python3
# Compare fp-ddp vs acados DDP solutions. Generic over state/control count.
#   python compare/compare.py [fpddp_file] [acados_file]
# Defaults to the 2-state pendulum files.
import sys


def load(path):
    d = {'traj': []}
    mode = None
    for line in open(path):
        s = line.split()
        if not s:
            continue
        if s[0] == 'traj':
            mode = 'traj'
            continue
        if mode == 'traj':
            d['traj'].append([float(v) for v in s])
        else:
            d[s[0]] = ' '.join(s[1:])
    return d


def main():
    fa = sys.argv[1] if len(sys.argv) > 1 else 'logs/cmp_fpddp.txt'
    fb = sys.argv[2] if len(sys.argv) > 2 else 'logs/cmp_acados.txt'
    try:
        a = load(fa)
        b = load(fb)
    except FileNotFoundError as e:
        print(f"missing result file: {e.filename}")
        print("generate it first (./build/cmp_* and compare/acados_*.py)")
        return 2

    n = min(len(a['traj']), len(b['traj']))
    if n == 0:
        print("empty trajectories")
        return 2
    ncol = min(len(a['traj'][0]), len(b['traj'][0]))
    dmax = [0.0] * ncol
    for i in range(n):
        for c in range(1, ncol):                       # column 0 is the index
            dmax[c] = max(dmax[c], abs(a['traj'][i][c] - b['traj'][i][c]))
    dc = abs(float(a['cost']) - float(b['cost']))

    print(f"problem: {a.get('problem','?')}")
    print(f"  fp-ddp : cost={a['cost']}  iters={a['iters']}")
    print(f"  acados : cost={b['cost']}  iters={b['iters']}")
    print("  max |Δ| per traj column (states then controls): " + "  ".join(f"{d:.2e}" for d in dmax[1:]))
    print(f"  |Δcost| = {dc:.3e}")
    tol = 1e-3
    ok = max(dmax[1:]) < tol
    print(f"  ==> {'AGREE' if ok else 'DIFFER'} (tol {tol:.0e})")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
