"""The dataset table and size macros (the paper's Table A1), in two steps.

    python3 code/tools/gen_datasets.py table --data data --out OUT/generated/datasets.tex

table reads data/dataset_rows.json, checked by its sha256 in data/SHA256SUMS, requires every prompt once per split
and each family's training and evaluation prompts disjoint (Appendix A says so), and writes the table and the four
size macros.

rows wrote data/dataset_rows.json from the files it names there, which carry prompt texts and are not in this pack:
the training rows the runs' estimator and routers were built from (their sha256 is the corpus the AvengersPro
artifacts record), and the evaluation prompts and the scored corpora the rendered campaign pins
(configs/render36/plan.json in the code repository: `datasets` and `scored_corpora`); an evaluation prompt's source
dataset is looked up in its family's scored corpus by the prompt text. Each row is the sha256 of its prompt text and
its source dataset, sorted; no prompt text, since the source datasets' licences restrict redistribution. Whoever
holds a source dataset can hash its prompts and find their rows.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

TRAIN_ROWS = {"qwen": ("qwen_train_rows_20260928.jsonl",
                       "0a624b7f4e88d53e494dd1af00c445283d2c01450bc88a94901f1818e7dcec96"),
              "gemma": ("gemma_train_rows_20261002.jsonl",
                        "3fd72d1b6853318b035f65b5db8987a79a82daa578ab039e3cf311cd4c7fbd47")}
EVAL = {"qwen": "eval_qwen", "gemma": "eval_gemma"}
NAMES = {"beaver_tails": "BeaverTails", "code_ultra_feedback": "CodeUltraFeedback", "gsm8k": "GSM8K",
         "lmsys": "LMSYS-Chat-1M", "mix_instruct": "MixInstruct", "reward_bench": "RewardBench", "squad": "SQuAD"}
FAMILIES, SPLITS = ("qwen", "gemma"), ("train", "eval")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rows(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def pinned(path: Path, want: str) -> Path:
    got = sha256(path.read_bytes())
    if got != want:
        raise SystemExit(f"{path}: sha256 {got}, the pin says {want}")
    return path


def entry(name: str, digest: str, pairs: list) -> dict:
    """A split: its file, and each row as "<sha256 of the prompt text> <dataset>", sorted."""
    return {"file": name, "sha256": digest,
            "rows": sorted(f"{sha256(p.encode())} {d}" for p, d in pairs)}


def build_rows(a) -> None:
    scope = json.loads(a.scope.read_text())
    doc = {}
    for fam in FAMILIES:
        name, digest = TRAIN_ROWS[fam]
        train = entry(name, digest, [(r["prompt"], r["dataset"]) for r in rows(pinned(a.inputs / name, digest))])
        corpus = scope["scored_corpora"][fam]
        by_prompt = {r["prompt"]: r["dataset"] for r in rows(pinned(a.data / corpus["path"], corpus["sha256"]))}
        ds = scope["datasets"][EVAL[fam]]
        prompts = rows(pinned(a.data / ds["path"], ds["sha256"]))
        missing = [p["id"] for p in prompts if p["prompt"] not in by_prompt]
        if missing:
            raise SystemExit(f"{fam}: {len(missing)} evaluation prompts are not in the scored corpus, e.g. {missing[:3]}")
        ev = entry(ds["path"], ds["sha256"], [(p["prompt"], by_prompt[p["prompt"]]) for p in prompts])
        ev["corpus"] = {"file": corpus["path"], "sha256": corpus["sha256"]}
        doc[fam] = {"train": train, "eval": ev}
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"{a.out}: " + ", ".join(f"{f} {s} {len(doc[f][s]['rows'])}" for f in FAMILIES for s in SPLITS))


def build_table(a) -> None:
    sums = {}
    for line in (a.data / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.strip().lstrip("*")] = digest
    raw = (a.data / "dataset_rows.json").read_bytes()
    if sha256(raw) != sums.get("dataset_rows.json"):
        raise SystemExit("dataset_rows.json: sha256 differs from data/SHA256SUMS")
    doc = json.loads(raw)
    counts = {}
    for fam in FAMILIES:
        prompts = {}
        for split in SPLITS:
            parsed = [r.split(" ") for r in doc[fam][split]["rows"]]
            if any(len(p) != 2 or len(p[0]) != 64 for p in parsed):
                raise SystemExit(f"{fam} {split}: a row is not '<sha256> <dataset>'")
            prompts[split] = {p[0] for p in parsed}
            if len(prompts[split]) != len(parsed):
                raise SystemExit(f"{fam} {split}: a prompt appears twice")
            counts[(fam, split)] = collections.Counter(p[1] for p in parsed)
        if prompts["train"] & prompts["eval"]:
            raise SystemExit(f"{fam}: {len(prompts['train'] & prompts['eval'])} prompts are in both splits, "
                             f"and Appendix A says the splits are disjoint")
    cols = [("qwen", "train"), ("qwen", "eval"), ("gemma", "train"), ("gemma", "eval")]
    unknown = set().union(*counts.values()) - set(NAMES)
    if unknown:
        raise SystemExit(f"datasets without a display name: {sorted(unknown)}")

    def n(x: int) -> str:
        return f"{x:,}".replace(",", "{,}")

    out = ["% Written by tools/gen_datasets.py from the pinned files; do not edit.",
           r"\newcommand{\qwenTrainPrompts}{" + n(sum(counts[("qwen", "train")].values())) + "}",
           r"\newcommand{\gemmaTrainPrompts}{" + n(sum(counts[("gemma", "train")].values())) + "}",
           r"\newcommand{\qwenEvalPrompts}{" + n(sum(counts[("qwen", "eval")].values())) + "}",
           r"\newcommand{\gemmaEvalPrompts}{" + n(sum(counts[("gemma", "eval")].values())) + "}",
           r"\newcommand{\datasetTable}{%",
           r"\begin{tabular}{lrrrr}", r"\toprule",
           r" & \multicolumn{2}{c}{Qwen2.5} & \multicolumn{2}{c}{Gemma 3} \\",
           r"Dataset & Train & Eval & Train & Eval \\", r"\midrule"]
    for key in sorted(NAMES, key=lambda k: NAMES[k].lower()):
        out.append(f"{NAMES[key]} & " + " & ".join(n(counts[c][key]) for c in cols) + r" \\")
    out += [r"\midrule", "Total & " + " & ".join(n(sum(counts[c].values())) for c in cols) + r" \\",
            r"\bottomrule", r"\end{tabular}}"]
    a.out.write_text("\n".join(out) + "\n")
    print(f"{a.out}: " + ", ".join(f"{f} {s} {sum(counts[(f, s)].values())}" for f, s in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("rows", help="the per-row file, from the pinned files")
    r.add_argument("--scope", type=Path, required=True)
    r.add_argument("--data", type=Path, required=True)
    r.add_argument("--inputs", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("table", help="the table and the size macros, from data/dataset_rows.json")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build_rows(a) if a.step == "rows" else build_table(a)


if __name__ == "__main__":
    main()
