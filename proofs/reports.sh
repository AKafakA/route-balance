#!/usr/bin/env bash
# The four reports rebuilt by one code tree from the archived runs, every file and log compared with the run bundle's.
#
#   proofs/reports.sh TREE ADAPTER BUNDLE PYTHON OUT
#
# TREE      the code tree (its analysis/, gates/, tools/ and configs/ are used); copied into OUT/code
# ADAPTER   submission or ref: the tree's parity gate is `python -m tools.cell_driver parity`
# BUNDLE    the run bundle: runs/<family>/<run>/, views/{q1,qall,g1,gall}/ (links to the runs), reports/report-*/ and
#           reports/report-*.log, inputs/ (the data roots, the SLO bases, the budget judge stores, the hosts file)
# PYTHON    the scheduler environment's python
# OUT       a new folder laid out as the bundle made its reports (code/, inputs, render36, views/, reports/), so every
#           path a report records is the bundle's
#
# The first-seed views are the bundle's: they hold no parity record, so both reports read every group "not run".
# The all-seeds views are scratch copies (a link to every run) with an empty _groups/; the tree's own parity gate runs
# over every comparison group there first, so the report reads each group from the tree's own parity records. Nothing
# is written into the bundle. Exit 0 only if every file of the four reports and every log equals the bundle's.
set -euo pipefail
TREE=$(cd "$1" && pwd) ADAPTER=$2 BUNDLE=$(cd "$3" && pwd) PY=$4 OUT=$5
[ ! -e "$OUT" ] || { echo "$OUT exists" >&2; exit 2; }
mkdir -p "$OUT/views" "$OUT/reports" "$OUT/tmp"
OUT=$(cd "$OUT" && pwd)
cp -r "$TREE" "$OUT/code"
ln -s "$BUNDLE/inputs" "$OUT/inputs"
ln -s code/configs/render36 "$OUT/render36"
ln -s "$BUNDLE/views/q1" "$OUT/views/q1"
ln -s "$BUNDLE/views/g1" "$OUT/views/g1"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=$OUT/code TMPDIR=$OUT/tmp

parity() {   # parity VIEW GROUP  (run in OUT/code)
  case "$ADAPTER" in
    submission|ref) "$PY" -m tools.cell_driver parity --render ../render36 --cells-root "$1" --group "$2" ;;
    *) echo "unknown adapter $ADAPTER" >&2; return 2 ;;
  esac
}

cd "$OUT/code"
for v in qall gall; do
  mkdir "$OUT/views/$v" "$OUT/views/$v/_groups"
  for e in "$BUNDLE/views/$v"/*; do
    [ "$(basename "$e")" = _groups ] || ln -s "$(readlink -f "$e")" "$OUT/views/$v/$(basename "$e")"
  done
done
for g in qwen qwen-ablation qwen-budget-30 qwen-budget-45 qwen-budget-75 qwen-sweep qwen-sweep2; do
  parity ../views/qall "$g" > "$OUT/views/parity-qall-$g.log" 2>&1 || { cat "$OUT/views/parity-qall-$g.log"; exit 1; }
done
parity ../views/gall gemma > "$OUT/views/parity-gall-gemma.log" 2>&1 || { cat "$OUT/views/parity-gall-gemma.log"; exit 1; }

report() {   # report VIEW FAMILY  (the bundle's own command)
  local v=$1 f=$2 fl store=()
  fl=$([ "$f" = qwen ] && echo qwen_std || echo gemma)
  [ "$f" = qwen ] && store=(--store ../inputs/qwen/budget-judge/store.s1.jsonl ../inputs/qwen/budget-judge/store.s2-zero.jsonl)
  "$PY" -m analysis.report --render ../render36 --cells "../views/$v" --data-root "../inputs/$f/data" \
    --fleet "$fl=configs/fleet_$f.json" --hosts ../inputs/hosts.lan.json --slo-base "$f=../inputs/$f/slo_base_$f.json" \
    "${store[@]}" --rules configs/prereg_rules.json --treatment RouteBalance-static --out "../reports/report-$v" \
    > "../reports/report-$v.log" 2>&1 || true
}
report q1 qwen & report qall qwen & report g1 gemma & report gall gemma & wait

: > "$OUT/compare.txt"
for v in q1 qall g1 gall; do
  if diff -r -q "$BUNDLE/reports/report-$v" "$OUT/reports/report-$v" > "$OUT/reports/diff-$v.txt" \
     && cmp -s "$BUNDLE/reports/report-$v.log" "$OUT/reports/report-$v.log"; then
    echo "report-$v: $(find "$OUT/reports/report-$v" -type f | wc -l) files and the log equal the bundle's" >> "$OUT/compare.txt"
  else
    echo "report-$v: DIFFERS ($(wc -l < "$OUT/reports/diff-$v.txt") files differ; log equal: $(cmp -s "$BUNDLE/reports/report-$v.log" "$OUT/reports/report-$v.log" && echo yes || echo no))" >> "$OUT/compare.txt"
  fi
done
cat "$OUT/compare.txt"
! grep -q DIFFERS "$OUT/compare.txt"
