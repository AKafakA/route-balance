"""The benchmark prompt files the campaign pins, without their text, and rebuilt byte for byte once the prompts are back.

    python -m tools.prompt_files extract --prompts-file FILE --labels LABELS.jsonl [...] --out MANIFEST.jsonl
    python -m tools.prompt_files build --manifest MANIFEST.jsonl --prompts PROMPTS.jsonl --out FILE

A pinned prompt file (configs/render36/plan.json `datasets`) holds one {"id", "prompt"} object per line, the evaluation
file of Qwen also its "source". extract writes its manifest: a first line {"file", "sha256", "rows", "ensure_ascii"}
(the file's pin and how its lines were serialized), then per line its id, its source when the file has one, the sha256
of its prompt, and the prompt's dataset (looked up in the label files by the prompt's sha256, or given by --datasets
for a prompt no label line holds). data/prompt_files/ holds the manifests of the four prompt files the paper's runs sent.

build writes the file again from the manifest and PROMPTS (tools/fetch_prompts.py's output, which takes the manifests as
it takes label files) and refuses a result whose sha256 is not the manifest's pin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def extract(a) -> int:
    datasets = {}
    for path in a.labels:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                datasets.setdefault(r["prompt_sha256"], r["dataset"])
    if a.datasets:
        datasets.update(json.load(open(a.datasets)))
    raw = open(a.prompts_file, "rb").read()
    lines = [line for line in raw.decode("utf-8").split("\n") if line]
    objs = [json.loads(line) for line in lines]
    ascii_ok = all(json.dumps(o) == line for o, line in zip(objs, lines))
    utf8_ok = all(json.dumps(o, ensure_ascii=False) == line for o, line in zip(objs, lines))
    if not (ascii_ok or utf8_ok):
        print(f"{a.prompts_file}: its lines are not json.dumps output", file=sys.stderr)
        return 1
    head = {"file": a.prompts_file.split("/")[-1], "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(objs),
            "ensure_ascii": ascii_ok}
    missing = 0
    with open(a.out, "x") as out:
        out.write(json.dumps(head) + "\n")
        for o in objs:
            if list(o) not in (["id", "prompt"], ["id", "prompt", "source"]):
                print(f"{a.prompts_file}: a line holds {list(o)}", file=sys.stderr)
                return 1
            sha = sha256_text(o["prompt"])
            if sha not in datasets:
                missing += 1
                continue
            line = {"id": o["id"], "prompt_sha256": sha, "dataset": datasets[sha]}
            if "source" in o:
                line["source"] = o["source"]
            out.write(json.dumps(line) + "\n")
    if missing:
        print(f"{missing} prompts have no dataset (give them with --datasets)", file=sys.stderr)
        return 1
    print(f"{a.out}: {len(objs)} lines of {head['file']}")
    return 0


def build(a) -> int:
    by_sha = {}
    with open(a.prompts) as f:
        for line in f:
            p = json.loads(line)["prompt"]
            by_sha[sha256_text(p)] = p
    with open(a.manifest) as f:
        head = json.loads(f.readline())
        rows = [json.loads(line) for line in f]
    out, missing = [], 0
    for r in rows:
        if r["prompt_sha256"] not in by_sha:
            missing += 1
            continue
        o = {"id": r["id"], "prompt": by_sha[r["prompt_sha256"]]}
        if "source" in r:
            o["source"] = r["source"]
        out.append(json.dumps(o, ensure_ascii=head["ensure_ascii"]) + "\n")
    if missing:
        print(f"{missing} prompts are not in {a.prompts}", file=sys.stderr)
        return 1
    raw = "".join(out).encode("utf-8")
    got = hashlib.sha256(raw).hexdigest()
    if got != head["sha256"] or len(out) != head["rows"]:
        print(f"the rebuilt {head['file']} has sha256 {got}, the pin says {head['sha256']}", file=sys.stderr)
        return 1
    with open(a.out, "xb") as f:
        f.write(raw)
    print(f"{a.out}: {len(out)} lines, sha256 {got} as pinned")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--prompts-file", required=True)
    ex.add_argument("--labels", nargs="+", required=True)
    ex.add_argument("--datasets", help="JSON {prompt sha256: dataset} for prompts no label line holds")
    ex.add_argument("--out", required=True)
    bu = sub.add_parser("build")
    bu.add_argument("--manifest", required=True)
    bu.add_argument("--prompts", required=True)
    bu.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return extract(a) if a.cmd == "extract" else build(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
