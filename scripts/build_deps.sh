#!/usr/bin/env bash
# Build + install the vendored blasfeo and hpipm submodules into external/install.
# Run once after `git submodule update --init`, before configuring fp-ddp.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$HERE/external/install"
JOBS="$(nproc 2>/dev/null || echo 4)"

cmake -S "$HERE/external/blasfeo" -B "$HERE/external/blasfeo/build" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBLASFEO_TARGET=GENERIC \
      -DBLASFEO_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$HERE/external/blasfeo/build" --target install -j"$JOBS"

cmake -S "$HERE/external/hpipm" -B "$HERE/external/hpipm/build" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBLASFEO_PATH="$PREFIX" \
      -DHPIPM_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$HERE/external/hpipm/build" --target install -j"$JOBS"

echo "blasfeo + hpipm installed to $PREFIX"
