"""The judged labels without any text, and the rows rebuilt from them once the prompts are back.

    python -m tools.labels extract --rows ROWS.jsonl --out LABELS.jsonl
    python -m tools.labels join --labels LABELS.jsonl --prompts PROMPTS.jsonl --out ROWS.jsonl

extract keeps, for every line of a scored corpus or training rows file: request_id, source (the public dataset's row id,
"<dataset>/<id>"), dataset, is_harmful when present, input_len, prompt_sha256 (the sha256 of the prompt's UTF-8 text,
as the paper data pack's dataset_rows.json) and, for every model, its output_length and its judged scores
(llm_judge_scores). The prompt, the reference answer and every answer's text are dropped: the datasets' licences restrict
redistributing them. data/labels/ holds the paper's labels made this way.

join puts the prompts back. PROMPTS is JSONL with a "prompt" on every line (other fields are ignored); each label line
takes the prompt whose sha256 is its prompt_sha256, and a label whose prompt is missing fails the join. The output keeps
the labels' order, in the format tools/knn_index.py, tools/router_training.py and the analysis read; no model carries a
"response", so BEST-Route's training data (which holds the answers' texts) still needs the answers collected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

TOP = ("request_id", "source", "dataset", "is_harmful", "input_len")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def extract(rows_path: str, out_path: str) -> int:
    n = 0
    with open(rows_path) as fin, open(out_path, "x") as fout:
        for line in fin:
            r = json.loads(line)
            lab = {k: r[k] for k in TOP if k in r}
            lab["prompt_sha256"] = sha256_text(r["prompt"])
            lab["models"] = {m: {"output_length": d.get("output_length"), "llm_judge_scores": d.get("llm_judge_scores") or {}}
                             for m, d in r["models"].items()}
            fout.write(json.dumps(lab) + "\n")
            n += 1
    print(f"{out_path}: {n} label lines")
    return 0


def join(labels_path: str, prompts_path: str, out_path: str) -> int:
    by_sha = {}
    with open(prompts_path) as f:
        for line in f:
            p = json.loads(line)["prompt"]
            by_sha[sha256_text(p)] = p
    n = missing = 0
    with open(labels_path) as fin, open(out_path, "x") as fout:
        for line in fin:
            lab = json.loads(line)
            prompt = by_sha.get(lab["prompt_sha256"])
            if prompt is None:
                missing += 1
                continue
            row = {k: lab[k] for k in ("request_id",) if k in lab}
            row["prompt"] = prompt
            row.update({k: lab[k] for k in TOP[1:] if k in lab})
            row["models"] = lab["models"]
            fout.write(json.dumps(row) + "\n")
            n += 1
    if missing:
        print(f"{missing} labels have no prompt in {prompts_path}", file=sys.stderr)
        return 1
    print(f"{out_path}: {n} rows")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--rows", required=True)
    ex.add_argument("--out", required=True)
    jo = sub.add_parser("join")
    jo.add_argument("--labels", required=True)
    jo.add_argument("--prompts", required=True)
    jo.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return extract(a.rows, a.out) if a.cmd == "extract" else join(a.labels, a.prompts, a.out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
