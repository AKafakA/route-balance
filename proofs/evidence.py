"""Every archived run's evidence checked by one code tree, against the run's own archived verdict.

    python -m proofs.evidence --adapter A --runs ROOT --render DIR --out FILE [--shard I/N] [--only RUN...]

For each run: the record and engine checks of its verdict, recomputed (the record and every evidence file parsed by
the tree's strict schema, the record against the render's scope, the engines unchanged, the prefix cache reset proven,
no traffic but the run's own), and the two run gates (arm, zero_call) on the run's directory. The run's archived
`gates/verdict.json` says what each gave when the run ran. One line per run: {run, problems, gates, verdict_problems,
verdict_gates, equal}; the summary on stderr counts the runs whose results equal their verdict's. Nothing is written
into the archive.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from proofs import archive as A

GATES = ("arm", "zero_call")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adapter", required=True)
    for name in ("runs", "render", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    adapter = importlib.import_module(f"proofs.adapters.{a.adapter}")
    i, n = (int(x) for x in a.shard.split("/"))
    todo = A.runs(a.runs, only=set(a.only) if a.only else None)[i::n]
    equal = 0
    with open(a.out, "a") as f:
        for run in todo:
            verdict = A.read_json(run / "gates" / "verdict.json")
            problems = adapter.verdict_problems(a.render, run)
            gates = {g: adapter.gate(g, run) for g in GATES}
            same = problems == verdict["bench_problems"] and {g: rc for g, (rc, _) in gates.items()} == verdict["gates"]
            equal += same
            A.write_line(f, {"run": run.name, "problems": problems, "gates": {g: rc for g, (rc, _) in gates.items()},
                             "gate_output": {g: out[-400:] for g, (rc, out) in gates.items() if rc != 0},
                             "verdict_problems": verdict["bench_problems"], "verdict_gates": verdict["gates"],
                             "verdict_valid": verdict["valid"], "equal": same})
    print(f"evidence: {len(todo)} runs, {equal} equal to their archived verdict", file=sys.stderr)
    return 0 if equal == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
