#!/usr/bin/env bash
# Build the acados C library (hpipm backend) for the DDP comparison.
set -euo pipefail
cd "$HOME/projects/acados"
git submodule update --init external/blasfeo external/hpipm
cmake -S . -B build \
  -DACADOS_WITH_OPENMP=OFF -DACADOS_EXAMPLES=OFF -DACADOS_UNIT_TESTS=OFF \
  -DBLASFEO_TARGET=GENERIC -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target install
echo "ACADOS_C_LIB_BUILT"
