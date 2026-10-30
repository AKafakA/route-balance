"""The training-row subsets behind the estimator ablations: a fraction of the labelled rows, or every dataset but one.

    python -m tools.row_subsets fraction --rows ROWS.jsonl --fraction 0.5 --out SUBSET.jsonl
    python -m tools.row_subsets without --rows ROWS.jsonl --dataset gsm8k --out SUBSET.jsonl

fraction keeps int(n * fraction) of the n rows, drawn by random.Random(20260731).sample and kept in file order: the
paper's indexes on 75, 50 and 25% of the labels. without keeps every row whose dataset is not the given one, in file
order: the leave-one-dataset-out indexes. ROWS is `router_training filter`'s output; each kept line is copied byte for
byte, and OUT must not exist. Each subset is indexed by tools/knn_index.py like the full rows.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

SEED = 20260731


def read_rows(path: str) -> list[str]:
    with open(path, newline="") as f:
        rows = [line for line in f.read().split("\n") if line]
    for line in rows:
        json.loads(line)
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("fraction")
    fr.add_argument("--fraction", required=True, type=float)
    wo = sub.add_parser("without")
    wo.add_argument("--dataset", required=True)
    for p in (fr, wo):
        p.add_argument("--rows", required=True)
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = read_rows(a.rows)
    if a.cmd == "fraction":
        if not 0 < a.fraction < 1:
            print("--fraction is between 0 and 1", file=sys.stderr)
            return 2
        keep = [rows[i] for i in sorted(random.Random(SEED).sample(range(len(rows)), int(len(rows) * a.fraction)))]
        what = f"{a.fraction:g} of the labels (seed {SEED}, file order kept)"
    else:
        keep = [line for line in rows if json.loads(line)["dataset"] != a.dataset]
        if len(keep) == len(rows):
            print(f"no rows of {a.dataset}", file=sys.stderr)
            return 1
        what = f"without {a.dataset}"
    with open(a.out, "x", newline="") as f:
        f.write("".join(line + "\n" for line in keep))
    print(f"{a.out}: {len(keep)} of {len(rows)} rows, {what}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
