#!/usr/bin/env bash
set -uo pipefail
SRC=${SRC:-$HOME/Code/llm/Block/block}
OUT=${OUT:-/tmp/rb_pub_build/route_balance}
rm -rf "$(dirname "$OUT")"; mkdir -p "$OUT"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pt' "$SRC/" "$OUT/"
rm -rf "$OUT/exp/end_to_end_exp_scripts" "$OUT"/cl_manifest_*.xml
# rename ANY dir whose name contains cara (deepest-first)
find "$OUT" -depth -type d -name '*cara*' | while read -r p; do
  nb=$(basename "$p" | sed 's/cara/route_balance/g'); mv "$p" "$(dirname "$p")/$nb"; done
# rename files containing cara
find "$OUT" -depth -type f -name '*cara*' | while read -r p; do
  nb=$(basename "$p" | sed 's/cara/route_balance/g'); mv "$p" "$(dirname "$p")/$nb"; done
# content transform
grep -rlI . "$OUT" 2>/dev/null | while read -r f; do
  sed -i -e 's/CARA_/ROUTE_BALANCE_/g' -e 's/CARA\([A-Z]\)/RouteBalance\1/g' -e 's/\bCARA\b/ROUTE_BALANCE/g' -e 's/Cara/RouteBalance/g' -e 's/cara/route_balance/g' \
         -e 's/\bblock\./route_balance./g' -e 's#\bblock/#route_balance/#g' "$f"
done
for f in $(grep -rEl 'hf_[A-Za-z0-9]{30,}' "$OUT" 2>/dev/null); do
  perl -i -pe 's/hf_[A-Za-z0-9]{30,}/\$\{HF_TOKEN\}/g' "$f"; done
echo "build: $(du -sh "$OUT" | cut -f1) | cara tokens (ci): $(grep -rIoi 'cara' "$OUT" 2>/dev/null | wc -l) | cara names: $(find "$OUT" -name '*cara*' | wc -l) | block. refs: $(grep -rIo '\bblock\.' "$OUT" 2>/dev/null | wc -l) | hf tokens: $(grep -rEl 'hf_[A-Za-z0-9]{30,}' "$OUT" 2>/dev/null | wc -l)"
