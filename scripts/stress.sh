#!/usr/bin/env bash
# Stress-compare fp-ddp against acados DDP on the shared problem set:
# build + run the fp-ddp side, codegen + run the acados side, diff per-iteration.
# Needs libacados built (scripts/build_acados.sh) and acados_template importable.
set -euo pipefail
cd "$(dirname "$0")/.."

cmake --build build -j"$(nproc)" >/dev/null
mkdir -p logs
./build/stress_fpddp all

export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$HOME/projects/acados}"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
python3 compare/stress_acados.py all
python3 compare/stress_compare.py
