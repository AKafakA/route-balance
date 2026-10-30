"""BEST-Route's training data in upstream's format (microsoft/best-route-llm, data.py), from the training rows.

    python -m tools.best_route_upstream --family qwen --rows ROWS.jsonl --split BR_DATA --out DIR

Every training row (`router_training filter`'s output) becomes one upstream example: the prompt as the instruction, the
family's reference model's answer as the output, and one candidate per model of the family's ladder, in ladder order,
with its answer, its judged score under the key upstream reads as the quality metric (armoRM_scores holds the DeepEval
score: the key is upstream's, the numbers are the judge's), the prompt's and the answer's token counts, and a cost of 0
(upstream recomputes the cost from the token counts and its own price table). A row without a prompt, or without an
answer from every model of the ladder, is skipped. The conversion is the one the paper's runs used, for both families.

DIR/_all.jsonl holds every example in row order. DIR/train.jsonl and DIR/val.jsonl split them by the ids of
`router_training best-route-data`'s split (BR_DATA/train.jsonl and BR_DATA/val.jsonl), which must cover the examples
exactly. DIR must not exist. The upstream training (scripts/train_best_route.sh) reads DIR/train.jsonl and DIR/val.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

JUDGE_KEY = "deepeval-llama3.1-8b-it_reference"
# (fleet model, upstream candidate base) in ladder order; upstream's candidate name is the base with "_bo1"
LADDERS = {
    "qwen": [("Qwen/Qwen2.5-3B", "Qwen2.5-3B"), ("Qwen/Qwen2.5-7B", "Qwen2.5-7B"),
             ("Qwen/Qwen2.5-14B", "Qwen2.5-14B"), ("Qwen/Qwen2.5-72B", "Qwen2.5-72B")],
    "gemma": [("google/gemma-3-1b-it", "gemma-3-1b"), ("google/gemma-3-4b-it", "gemma-3-4b"),
              ("google/gemma-3-12b-it", "gemma-3-12b"), ("google/gemma-3-27b-it", "gemma-3-27b")],
}
REFERENCE_MODEL = {"qwen": "Qwen/Qwen2.5-72B", "gemma": "google/gemma-3-27b-it"}   # the largest tier


def deepeval_score(model_data: dict) -> float:
    if model_data.get("deepeval_score") is not None:
        return float(model_data["deepeval_score"])
    judges = model_data.get("llm_judge_scores") or {}
    if JUDGE_KEY in judges:
        return float(judges[JUDGE_KEY])
    return 0.0


def convert(src_path: Path, dst_path: Path, family: str) -> tuple[int, int, int]:
    ladder, reference = LADDERS[family], REFERENCE_MODEL[family]
    n_in = n_out = n_skipped = 0
    with open(src_path) as fin, open(dst_path, "w") as fout:
        for line in fin:
            n_in += 1
            d = json.loads(line)
            prompt = d.get("prompt", "")
            models = d.get("models", {})
            if not prompt:
                n_skipped += 1
                continue
            input_len = int(d.get("input_len") or len(prompt.split()))
            cands, ok = [], True
            for src_name, base_name in ladder:
                m = models.get(src_name)
                if m is None:
                    ok = False
                    break
                resp = m.get("response", "")
                cands.append({
                    "text": resp,
                    "model": f"{base_name}_bo1",
                    "scores": {"armoRM_scores": deepeval_score(m)},
                    "token_num_prompt": input_len,
                    "token_num_responses": int(m.get("output_length") or len(resp.split())),
                    "cost": 0.0,
                })
            if not ok or len(cands) != len(ladder):
                n_skipped += 1
                continue
            fout.write(json.dumps({
                "id": d.get("request_id") or d.get("id") or f"row-{n_in}",
                "instruction": prompt,
                "input": "",
                "output": models.get(reference, {}).get("response", ""),
                "candidates": cands,
            }) + "\n")
            n_out += 1
    return n_in, n_out, n_skipped


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True, choices=sorted(LADDERS))
    ap.add_argument("--rows", required=True, type=Path, help="router_training filter's training rows")
    ap.add_argument("--split", required=True, type=Path, help="router_training best-route-data's output folder")
    ap.add_argument("--out", required=True, type=Path, help="a new folder")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    n_in, n_out, n_skipped = convert(a.rows, a.out / "_all.jsonl", a.family)
    print(f"{a.family}: {n_out}/{n_in} converted ({n_skipped} skipped); ladder {[b for _, b in LADDERS[a.family]]}")
    val_ids = {json.loads(line)["id"] for line in open(a.split / "val.jsonl")}
    train_ids = {json.loads(line)["id"] for line in open(a.split / "train.jsonl")}
    rows = [line for line in open(a.out / "_all.jsonl").read().split("\n") if line]
    ids = [json.loads(line)["id"] for line in rows]
    if set(ids) != val_ids | train_ids or val_ids & train_ids:
        print("the split's ids do not cover the examples exactly", file=sys.stderr)
        return 1
    with open(a.out / "train.jsonl", "x") as t, open(a.out / "val.jsonl", "x") as v:
        for i, line in zip(ids, rows):
            (v if i in val_ids else t).write(line + "\n")
    print(f"{a.family}: train {len(train_ids)}, val {len(val_ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
