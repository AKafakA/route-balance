"""Split merged corpora into judge shards of ROWS rows, in file order: shard_<prefix><k>.jsonl (prefix e for evaluation, t for
training, so the evaluation shards are judged first).

    python -m tools.judge_shards OUT_DIR ROWS PREFIX=CORPUS.jsonl [PREFIX=CORPUS.jsonl ...]

The corpora are tools/collect_answers.py merge's unscored output; the paper's judging used ROWS = 200, e= for the
evaluation corpus and t= for the training corpus. scripts/judge_pool.sh scores the shards.
"""
import sys
from pathlib import Path

out, rows = Path(sys.argv[1]), int(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
for spec in sys.argv[3:]:
    prefix, path = spec.split("=", 1)
    lines = [l + b"\n" for l in open(path, "rb").read().split(b"\n") if l]   # newlines only: a JSON string may hold U+2028 etc.
    for k in range(0, len(lines), rows):
        with open(out / f"shard_{prefix}{k // rows:03d}.jsonl", "xb") as f:
            f.writelines(lines[k:k + rows])
    print(f"{path}: {len(lines)} rows -> {(len(lines) + rows - 1) // rows} shards")
