#!/usr/bin/env bash
# Rebuilds the fixture archives from the encoders of past commits: the main
# branch after each pull request that changed ultracodec. Each commit's
# ultracodec/ is exported (git archive), its library built on its own, and
# make_fixture.cpp linked against it. Today's encoder is never used, so the
# archives are what those builds wrote; test_fixtures.cpp decodes them.
#
#   ultracodec/tests/fixtures/make_fixtures.sh [WORKDIR]
#
# Writes MANIFEST and the archives next to this script: <commit>-t<tier>[-variant].u
# for each archive not byte-identical to one an earlier commit wrote (MANIFEST
# lists every commit, tier and variant, and the file holding its archive).
# Encoders are deterministic, so a rerun on any machine should reproduce
# MANIFEST exactly; a difference is worth investigating, not committing.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
work=${1:-$(mktemp -d)}
cxx=${CXX:-c++}

# commit  pull request  what it changed for the format
commits=(
  "2b63b18 #7  first 4-tier engine"
  "334e2b3 #8  Tier 4 snapshot warm-start (SWS)"
  "e8a41de #10 Tier 1 entropy mode, Tier 2 word filter, Tier 4 layouts and bypass"
  "43005cf #12 Tier 4 int16 mixer"
  "7f3ee6e #13 compact Tier 3"
)

manifest="$here/MANIFEST"
{
  echo "# file  commit  tier  variant  bytes  sha256 (written by make_fixtures.sh; input: fixture_input.hpp)"
  echo "# A file named after an earlier commit: this commit wrote the same bytes."
} > "$manifest.new"
for entry in "${commits[@]}"; do
  read -r commit pr _ <<< "$entry"
  src="$work/$commit"
  mkdir -p "$src"
  git -C "$repo" archive "$commit" ultracodec | tar -x -C "$src"
  cmake -S "$src/ultracodec" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DULTRA_BUILD_TESTS=OFF > /dev/null
  cmake --build "$src/build" --target ultracodec -j > /dev/null
  defs=()
  variants=("")
  if [ "$commit" = 7f3ee6e ]; then
    defs=(-DUF_VARIANTS)
    variants=("" t1-raw t4-bypass t4-tables-5)
  fi
  "$cxx" -std=c++20 -O2 "${defs[@]}" -I"$here" -I"$src/ultracodec/include" "$here/make_fixture.cpp" \
    "$src/build/libultracodec.a" -pthread -o "$src/make_fixture"
  for tier in 1 2 3 4; do
    for v in "${variants[@]}"; do
      case "$v" in
        t1-*) [ "$tier" = 1 ] || continue ;;
        t4-*) [ "$tier" = 4 ] || continue ;;
      esac
      name="$commit-t$tier${v:+-${v#t?-}}.u"
      "$src/make_fixture" "$tier" "$work/$name" $v
      size=$(wc -c < "$work/$name")
      sum=$(sha256sum "$work/$name" | cut -d' ' -f1)
      same=$(awk -v s="$sum" '$6 == s { print $1; exit }' "$manifest.new")
      if [ -n "$same" ]; then
        name=$same
      else
        cp "$work/$name" "$here/$name"
      fi
      echo "$name  $commit  $tier  ${v:--}  $size  $sum" >> "$manifest.new"
      echo "$pr $commit tier $tier ${v:-default}: $size bytes -> $name"
    done
  done
done
mv "$manifest.new" "$manifest"
