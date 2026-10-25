#!/usr/bin/env bash
# Profile-guided build of hce (see HCE_PGO in CMakeLists.txt):
#
#   tools/pgo.sh [BUILD_DIR] [TRAINING_INPUT...]
#
# 1. configures BUILD_DIR-gen with HCE_PGO=GENERATE and builds hce,
# 2. compresses and decompresses the training inputs (default: this
#    repository's C++ sources) with both codecs at levels 0, 2 and 4, one
#    thread and four, and checks every round trip,
# 3. configures BUILD_DIR (default build/pgo) with HCE_PGO=USE and builds.
#
# CXX, CMAKE_BUILD_TYPE and extra cmake arguments (CMAKE_ARGS) pass through.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-$repo/build/pgo}
shift || true
gen=$build-gen
profile=$build-profile
type=${CMAKE_BUILD_TYPE:-Release}
read -r -a extra <<<"${CMAKE_ARGS:-}"

rm -rf "$profile"
cmake -S "$repo" -B "$gen" -G Ninja -DCMAKE_BUILD_TYPE="$type" -DHCE_BUILD_TESTS=OFF \
  -DHCE_PGO=GENERATE -DHCE_PGO_DIR="$profile" "${extra[@]}"
cmake --build "$gen" --target hce

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
if [ $# -gt 0 ]; then
  cat "$@" >"$work/train"
else
  find "$repo/codec" "$repo/hc" "$repo/encoder" "$repo/decoder" "$repo/structural" "$repo/cli" \
    -name '*.[ch]pp' -print0 | sort -z | xargs -0 cat >"$work/train"
fi
for level in 0 2 4; do
  for threads in 1 4; do
    for mode in "" "--structural --lang cpp"; do
      # shellcheck disable=SC2086
      "$gen/hce" -f -T "$threads" -l "$level" $mode -i "$work/train" -o "$work/z"
      "$gen/hce" -d -f -T "$threads" -i "$work/z" -o "$work/out"
      cmp "$work/train" "$work/out"
    done
  done
done

if [[ "${CXX:-}" == *clang* ]]; then
  llvm-profdata merge -o "$profile/hce.profdata" "$profile"/*.profraw
fi
cmake -S "$repo" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE="$type" -DHCE_PGO=USE -DHCE_PGO_DIR="$profile" \
  "${extra[@]}"
cmake --build "$build"
echo "pgo: profile-guided build in $build"
