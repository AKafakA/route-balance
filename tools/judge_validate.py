"""Validate a scorer output against its input. Exit 1 on any violation (fail loud).
Checks: same row count/order/request_id; every non-judge field identical; all pre-existing
llm_judge_scores keys identical; the new key present for every (row, model) scored, value in [0,1],
not None. Reports coverage, per-model mean, value histogram.
Exit codes: 0 = clean; 1 = schema/field/range error (hard failure); 2 = schema clean but some pairs
have no score (None) -- the full list is written to <output>.missing.json.

    python -m tools.judge_validate UNSCORED.jsonl SCORED.jsonl [MAX_ROWS]"""
import json, sys, collections
K = "deepeval-llama3.1-8b-it_reference"
inp, out = sys.argv[1], sys.argv[2]
max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 0
I = [json.loads(l) for l in open(inp)]
if max_rows: I = I[:max_rows]
O = [json.loads(l) for l in open(out)]
errs = []
if len(I) != len(O): errs.append(f"row count {len(I)} != {len(O)}")
missing = []; vals = collections.defaultdict(list); hist = collections.Counter()
for i, (a, b) in enumerate(zip(I, O)):
    if a.get("request_id") != b.get("request_id"): errs.append(f"row {i} request_id mismatch"); continue
    if {k: v for k, v in a.items() if k != "models"} != {k: v for k, v in b.items() if k != "models"}:
        errs.append(f"row {i} top-level fields changed")
    if set(a["models"]) != set(b["models"]): errs.append(f"row {i} model set changed")
    for m in a["models"]:
        ma, mb = a["models"][m], b["models"][m]
        if {k: v for k, v in ma.items() if k != "llm_judge_scores"} != {k: v for k, v in mb.items() if k != "llm_judge_scores"}:
            errs.append(f"row {i} {m} model fields changed")
        ja, jb = dict(ma.get("llm_judge_scores") or {}), dict(mb.get("llm_judge_scores") or {})
        v = jb.pop(K, None); ja.pop(K, None)
        if ja != jb: errs.append(f"row {i} {m} pre-existing judge keys changed")
        if v is None: missing.append((i, a.get("request_id"), m)); continue
        if not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0): errs.append(f"row {i} {m} bad score {v!r}")
        vals[m].append(v); hist[round(v, 4)] += 1
n = sum(len(v) for v in vals.values()); tot = n + len(missing)
print(f"rows {len(O)} | scored {n}/{tot} | missing(None) {len(missing)} | errors {len(errs)}")
for m in sorted(vals): print(f"  {m}: n={len(vals[m])} mean={sum(vals[m])/len(vals[m]):.4f}")
print("  value histogram:", dict(sorted(hist.items())))
for e in errs[:20]: print("  ERROR", e)
for x in missing[:20]: print("  MISSING", x)
if missing:
    json.dump(missing, open(out + ".missing.json", "w"))
    print(f"  missing list ({len(missing)}) -> {out}.missing.json")
sys.exit(1 if errs else (2 if missing else 0))
