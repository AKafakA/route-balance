#!/usr/bin/env bash
# Every number, table and figure of the paper, regenerated from data/ with the pack's own code, and compared with
# expected/ byte for byte.
#
#   bash reproduce_results.sh /path/to/new-output      (PY=<python> to choose the interpreter; default python3)
#
# Exit 0 only when the pack matches its manifest and every regenerated output equals its reference.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"      # the physical path: symlinks resolved
OUT="${1:?usage: bash reproduce_results.sh /path/to/new-output}"
PY="${PY:-python3}"
export PYTHONDONTWRITEBYTECODE=1          # the pack stays as shipped: no __pycache__ inside it
case "$PY" in */*) case "$PY" in /*) ;; *) PY="$PWD/$PY";; esac;; esac
case "$HERE" in *[[:space:]]*) echo "unpack the pack into a path without whitespace" >&2; exit 2;; esac
"$PY" -c 'import sys; assert sys.version_info[:2] >= (3, 10), sys.version' || { echo "Python 3.10 or later" >&2; exit 2; }
OUT="$("$PY" -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$OUT")"    # relative to the caller, symlinks resolved
[ ! -e "$OUT" ] || { echo "$OUT exists: give a new folder" >&2; exit 2; }
case "$OUT/" in "$HERE"/*) echo "give an output folder outside the pack" >&2; exit 2;; esac

cd "$HERE"
"$PY" verify_results.py manifest
mkdir -p "$OUT/generated" "$OUT/tables" "$OUT/figures"

# the numbers and tables (generated/results.tex, generated/plot_points.json, tables/gen_*.tex)
PYTHONPATH="$HERE/code" "$PY" code/tools/gen_results.py --data data --out "$OUT"

# the figures, from the points the generator wrote
P="$OUT/generated/plot_points.json"
"$PY" code/tools/plot_frontier3d.py "$P" "$OUT/figures/frontier3d.pdf"
"$PY" code/tools/plot_frontier.py "$P" "$OUT/figures/frontier2x6.pdf" --rows 2
"$PY" code/tools/plot_variants.py "$P" "$OUT/figures/variants.pdf"
"$PY" code/tools/plot_frontier.py "$P" "$OUT/figures/frontier_gemma.pdf" --family gemma
"$PY" code/tools/plot_frontier.py "$P" "$OUT/figures/tail_qwen.pdf" --p99
"$PY" code/tools/plot_frontier.py "$P" "$OUT/figures/tail_gemma.pdf" --p99 --family gemma

"$PY" verify_results.py compare "$OUT"
