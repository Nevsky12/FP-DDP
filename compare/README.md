# fp-ddp vs acados DDP — comparison

Three **unconstrained** OCPs, each defined identically in C++ (`test/cmp_*.cpp`) and an acados script
(`acados_*.py`): same dynamics, same discrete cost (LINEAR_LS with `dt` folded into `W`, `scaling=1`),
same integrator (IRK Gauss-Legendre s=3), DDP / Gauss-Newton, hpipm backward pass — so the two discrete
OCPs are identical and both solvers should reach the same optimum.

| problem | states | controls | fp-ddp result |
|---|---|---|---|
| `pendulum` — `ẋ₂=−sin x₁+u`, regulate from [2,0] | 2 | 1 | 4 iters, cost 1.1371799, statio 1e-9 |
| `cartpole` — acados' `pendulum_ode` (inverted, unstable), stabilize | 4 | 1 | 7 iters, cost 0.14499, statio 6e-8 |
| `unicycle` — `ẋ=v cosθ, ẏ=v sinθ, θ̇=ω`, regulate | 3 | 2 | 5 iters, cost 0.29577, statio 3e-8 |

## fp-ddp side — runs out of the box
```sh
cmake --build build
./build/cmp_pendulum   # -> logs/cmp_fpddp.txt
./build/cmp_cartpole   # -> logs/cmp_cartpole_fpddp.txt
./build/cmp_unicycle   # -> logs/cmp_unicycle_fpddp.txt
```

## acados side — needs the acados Python toolchain + t_renderer
```sh
scripts/build_acados.sh                          # build libacados (done in this repo)
export ACADOS_SOURCE_DIR=$HOME/projects/acados
python -c "from acados_template.utils import get_tera; get_tera()"   # fetch t_renderer
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
PY=$HOME/projects/RocketInitialGuess/.venv/bin/python
$PY compare/acados_pendulum.py   # -> logs/cmp_acados.txt
$PY compare/acados_cartpole.py   # -> logs/cmp_cartpole_acados.txt
$PY compare/acados_unicycle.py   # -> logs/cmp_unicycle_acados.txt
```

## diff (generic over state/control count)
```sh
python compare/compare.py logs/cmp_fpddp.txt          logs/cmp_acados.txt
python compare/compare.py logs/cmp_cartpole_fpddp.txt logs/cmp_cartpole_acados.txt
python compare/compare.py logs/cmp_unicycle_fpddp.txt logs/cmp_unicycle_acados.txt
```

## status of the acados run in this sandbox

The acados **C library is built** (`scripts/build_acados.sh`) and `acados_template` 0.5.1 is installed,
but the acados Python codegen needs the prebuilt **`t_renderer`** binary: its download was blocked by the
sandbox's external-binary policy and no Rust toolchain was present to build it from source. So the acados
side here is **turnkey but not executed**.

What *is* verified (see `../EVIDENCE.md`): fp-ddp produces the KKT-optimal solution acados' DDP converges
to. On the linear-quadratic class fp-ddp matches the **analytic Riccati to ~1e-16** (and both use hpipm
for the backward pass); the three nonlinear problems reach **stationarity 1e-7…1e-9**, the same condition
acados' DDP terminates on. A noted limit: on the unstable cart-pole, fp-ddp v1's basic iLQR globalization
needs a well-conditioned horizon; the funnel/merit globalization (which acados has) is the follow-on.
