"""A benchmark prompt file (bench/dataset.py: one {"id", "prompt"} per line) from a scored corpus.

The corpora carry each prompt as {"request_id": str, "prompt": str, ...}. The output keeps their file order and
exactly those two fields (request_id becomes id), so the same corpus always gives the same bytes, and the
scope pins the output by sha256. Prompts in any --exclude file (e.g. the evaluation set, when writing the
training split for W1 and W1b) are refused, not dropped: a held-out set that overlaps the evaluation set is a
defect of the inputs. Every id and prompt must be unique and every prompt non-empty (bench/dataset.py reads
the output back and checks the same).

Usage: python -m tools.prompt_set --corpus TRAIN.jsonl [--exclude EVAL.jsonl ...] --out PROMPTS.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bench.schema import read_jsonl
from bench.dataset import file_sha256, read_rows
from routebalance.config import ConfigError


def corpus_rows(path: Path) -> list[tuple[str, str]]:
    rows = []
    for lineno, rec in read_jsonl(path):
        if not isinstance(rec, dict) or not isinstance(rec.get("request_id"), str) \
                or not isinstance(rec.get("prompt"), str) or not rec["prompt"]:
            raise ConfigError(f"{path}:{lineno}: needs a string request_id and a non-empty string prompt")
        rows.append((rec["request_id"], rec["prompt"]))
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[], help="prompt files whose prompts must not appear")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise ConfigError(f"{a.out} exists; write a new file")
    rows = corpus_rows(a.corpus)
    excluded = {p for path in a.exclude for _, p in read_rows(path)}
    overlap = sum(1 for _, p in rows if p in excluded)
    if overlap:
        raise ConfigError(f"{overlap} prompts of {a.corpus} are in the excluded sets")
    for what, values in (("id", [r[0] for r in rows]), ("prompt", [r[1] for r in rows])):
        if len(set(values)) != len(values):
            raise ConfigError(f"{a.corpus}: duplicate {what} values")
    text = "".join(json.dumps({"id": i, "prompt": p}, ensure_ascii=False) + "\n" for i, p in rows)
    a.out.write_text(text, encoding="utf-8")
    if [r for r in read_rows(a.out)] != rows:
        raise ConfigError(f"{a.out} does not read back as written")
    print(f"{a.out}: {len(rows)} prompts, sha256 {file_sha256(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
