"""The judge loop for the budget sweep's served texts: collect every text that has no score yet, and merge
the judge's scores back into the served-text score store (analysis/quality.py ScoreStore).

    python -m tools.judge_served collect --judge-key K --store STORE.jsonl... --cells ROOT... --render DIR
                                         --data-root DIR --family qwen --out JUDGE_INPUT.jsonl
    python -m tools.judge_served merge --judge-key K --store STORE.jsonl... --judge-input JUDGE_INPUT.jsonl
                                       --judge-output JUDGE_OUTPUT.jsonl --out NEW_STORE.jsonl

A budget cell's quality is the judged score of the text it actually served: a reply cut short by its budget is
judged as served (every other cell's quality is the lookup of its prompt and model). collect reads every
successful request of every budget cell (a record with a budget table) under the cells roots
(<root>/<cell_id>/<cell_id>.cell.json); other cells are skipped and counted. A text is a (prompt, model,
sha256(text)) key; one already scored under the judge key is skipped, and each remaining key is written once. An
empty text is not sent (the judge scores no empty response): the store scores it 0. The
prompt's reference answer comes from the scored corpora the render's plan pins (scored_corpora.<family>: path
under the data root and sha256, checked); the judge is reference-grounded, and a prompt without a reference
answer fails the collection.

The judge is the DeepEval G-Eval scorer (tools/geval_score.py, run by scripts/judge_one.sh; Llama-3.1-8B-Instruct
served by vLLM, judge key deepeval-llama3.1-8b-it_reference). Its input is JSONL, one entry per line:
  {"request_id": str, "prompt": str, "reference_text": str,
   "models": {<name>: {"response": str, ...}, ...}, ...}
It scores every model of an entry with a non-empty response and a non-empty reference_text (prompt, response and
reference each truncated to 2048 characters), takes the model names from the FIRST entry only, and writes the
same entries in the same order with models.<name>.llm_judge_scores.<judge key> = a score in [0, 1] added where
the metric returned one (the rest are listed in <output>.failures.json). Every other field comes back unchanged.
So each input line here holds exactly one text under the fixed model name "served" (the served model and the
text hash ride along in its fields): every line is scored whatever model served it.

merge checks the judge's output line by line against the input (count, order, request_id and every field but
the score) and writes the scored texts to a new store file: never an existing one, never a key already in the
store under this judge key. Texts the judge did not score stay unscored and are listed.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Sequence

from bench.dataset import file_sha256
from analysis.quality import parse_store_line, text_sha256
from bench.schema import read_jsonl, write_new_jsonl
from bench.records import CELL_SUFFIX, read_cell, records_in
from routebalance.config import ConfigError
from tools.campaign import load_plan

SLOT = "served"


def store_keys(paths: Sequence[Path], judge_key: str) -> set[tuple[str, str, str]]:
    """Keys that already have a score under the judge key."""
    keys: set[tuple[str, str, str]] = set()
    for path in paths:
        for lineno, rec in read_jsonl(path):
            key, _, scores = parse_store_line(rec, f"{path}:{lineno}")
            if judge_key in scores:
                if key in keys:
                    raise ConfigError(f"{path}:{lineno}: this text already has a {judge_key} score in the store")
                keys.add(key)
    return keys


def served_texts(roots: Sequence[Path]) -> tuple[list[tuple[str, str, str, str]], int]:
    """(prompt, model, text, where) of every successful request of every budget cell under the roots, and how
    many other cells were skipped."""
    out, skipped = [], 0
    for root in roots:
        for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            names = records_in(d)
            if not names:
                continue
            if names != [f"{d.name}{CELL_SUFFIX}"]:
                raise ConfigError(f"{d}: a cell directory holds exactly its own record, found {names}")
            cell = read_cell(d / names[0])
            if cell.budgets_sha256 is None:
                skipped += 1
                continue
            out += [(r.prompt, r.reply.model, r.reply.generated_text, f"{cell.cell_id}#{r.index}")
                    for r in cell.requests if r.reply is not None]
    return out, skipped


def references(paths: Sequence[Path]) -> dict[str, str]:
    """prompt -> reference answer, from the scored corpora."""
    out: dict[str, str] = {}
    for path in paths:
        for lineno, rec in read_jsonl(path):
            if not isinstance(rec, dict) or not isinstance(rec.get("prompt"), str):
                raise ConfigError(f"{path}:{lineno}: a corpus line needs a string prompt")
            ref = rec.get("reference_text")
            if not isinstance(ref, str) or not ref:
                continue                      # a prompt without a reference fails only if a served text needs it
            if rec["prompt"] in out and out[rec["prompt"]] != ref:
                raise ConfigError(f"{path}:{lineno}: the prompt has two different reference answers")
            out[rec["prompt"]] = ref
    return out


def pinned_corpora(plan: dict, families: Sequence[str], data_root: Path) -> list[Path]:
    """The reference corpora of the families, refused unless each file has the plan's sha256."""
    out = []
    for family in families:
        if family not in plan["scored_corpora"]:
            raise ConfigError(f"the plan has no scored corpus for {family!r}")
        entry = plan["scored_corpora"][family]
        path = Path(data_root) / entry["path"]
        sha = file_sha256(path)
        if sha != entry["sha256"]:
            raise ConfigError(f"{path} has sha256 {sha}; the plan pins {entry['sha256']}")
        out.append(path)
    return out


def line_id(key: tuple[str, str, str]) -> str:
    return hashlib.sha256("\x00".join(key).encode("utf-8")).hexdigest()[:24]


def collect(judge_key: str, store: Sequence[Path], cells: Sequence[Path],
            corpora: Sequence[Path]) -> tuple[list[dict], int, int]:
    """(judge input lines, empty texts not sent, cells skipped as not budget cells)."""
    scored = store_keys(store, judge_key)
    refs = references(corpora)
    lines, seen = [], set()
    empty = 0
    missing_ref: list[str] = []
    served, skipped = served_texts(cells)
    for prompt, model, text, where in served:
        key = (prompt, model, text_sha256(text))
        if key in scored or key in seen:
            continue
        seen.add(key)
        if not text:
            empty += 1                                   # scored 0 by the store, never sent
            continue
        if prompt not in refs:
            missing_ref.append(where)
            continue
        lines.append({"request_id": line_id(key), "prompt": prompt, "reference_text": refs[prompt],
                      "models": {SLOT: {"response": text, "served_model": model, "text_sha256": key[2]}}})
    if missing_ref:
        raise ConfigError(f"{len(missing_ref)} texts have a prompt with no reference answer in the corpora, "
                          f"e.g. {missing_ref[:3]}")
    return lines, empty, skipped


def merge(judge_key: str, store: Sequence[Path], judge_input: Path, judge_output: Path) -> tuple[list[dict], list[str]]:
    """(new store lines, request ids the judge did not score)."""
    scored = store_keys(store, judge_key)
    sent = [rec for _, rec in read_jsonl(judge_input)]
    back = [rec for _, rec in read_jsonl(judge_output)]
    if len(sent) != len(back):
        raise ConfigError(f"the judge returned {len(back)} lines for {len(sent)}")
    new, unscored = [], []
    for n, (a, b) in enumerate(zip(sent, back, strict=True), start=1):
        where = f"{judge_output}:{n}"
        if not isinstance(b, dict) or set(b) != set(a) or any(b[k] != a[k] for k in a if k != "models"):
            raise ConfigError(f"{where}: the judge changed the line (its request_id or fields)")
        if set(b["models"]) != {SLOT}:
            raise ConfigError(f"{where}: the judge changed the model slot")
        sa, sb = a["models"][SLOT], b["models"][SLOT]
        extra = set(sb) - set(sa)
        if any(sb[k] != sa[k] for k in sa) or not extra <= {"llm_judge_scores"}:
            raise ConfigError(f"{where}: the judge changed the served text's fields")
        judged = sb["llm_judge_scores"] if "llm_judge_scores" in sb else {}
        if not isinstance(judged, dict) or not set(judged) <= {judge_key}:
            raise ConfigError(f"{where}: unexpected judge scores {judged!r}")
        if judge_key not in judged or judged[judge_key] is None:
            unscored.append(a["request_id"])
            continue
        key = (a["prompt"], sa["served_model"], sa["text_sha256"])
        if key in scored:
            raise ConfigError(f"{where}: this text already has a {judge_key} score in the store")
        line = {"prompt": key[0], "model": key[1], "text_sha256": key[2], "text": sa["response"],
                "scores": {judge_key: judged[judge_key]}}
        parse_store_line(line, where)                       # the score is a number in [0, 1], the hash the text's
        scored.add(key)
        new.append(line)
    return new, unscored


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    m = sub.add_parser("merge")
    for p in (c, m):
        p.add_argument("--judge-key", required=True)
        p.add_argument("--store", type=Path, nargs="*", required=True,
                       help="the store files so far (the first pass gives --store with no file)")
        p.add_argument("--out", type=Path, required=True)
    c.add_argument("--cells", type=Path, nargs="+", required=True, help="cells roots")
    c.add_argument("--render", type=Path, required=True, help="the campaign render whose plan pins the corpora")
    c.add_argument("--data-root", type=Path, required=True)
    c.add_argument("--family", nargs="+", required=True, help="the families whose reference answers to load")
    m.add_argument("--judge-input", type=Path, required=True)
    m.add_argument("--judge-output", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        corpora = pinned_corpora(load_plan(a.render), a.family, a.data_root)
        lines, empty, skipped = collect(a.judge_key, a.store, a.cells, corpora)
        write_new_jsonl(a.out, lines)
        print(f"{a.out}: {len(lines)} texts to judge under {a.judge_key}; {empty} empty served texts (scored 0, not "
              f"sent); {skipped} cells without a budget skipped (their quality is the lookup)")
        return 0
    new, unscored = merge(a.judge_key, a.store, a.judge_input, a.judge_output)
    write_new_jsonl(a.out, new)
    print(f"{a.out}: {len(new)} scored texts added; {len(unscored)} left unscored by the judge")
    return 0 if not unscored else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
