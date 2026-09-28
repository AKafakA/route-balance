"""Every model's answer to every prompt of a prompt set, directly on the engines, and the scored corpus rebuilt from
them (D102: the Qwen answers re-collected once every request stops at the end of its turn).

    python -m tools.collect_answers collect --render DIR --fleet-name NAME --fleet FLEET.json --hosts HOSTS.json
                                            --data-root DIR --dataset NAME --concurrency C --out ANSWERS.jsonl
    python -m tools.collect_answers merge --corpus SCORED.jsonl --answers ANSWERS.jsonl --out NEW.jsonl

collect: the render's dataset (bytes and row count checked) is answered once by every model of the render's fleet
(the file checked byte for byte, tools/direct.py). A model's prompts are split across its instances in fleet order
(prompt i to instance i mod n) and each instance runs a closed loop of C requests (tools/direct.py closed_loop,
no fillers) through the scheduler's backend client, so a request is sent exactly as a served one: the raw prompt
as the single user message, templated once by vLLM, the scope's sampling, the tier's stop tokens. The engines are
drained and warmed up first and drained at the end. One line per prompt and model, written when every instance
has finished or failed: prompt_id, row, prompt, model, instance_id, and the backend outcome (text, prompt_tokens,
output_tokens, success, error, ...). A failed request or a failed instance fails the run (exit 1), after the finished
lines are written.

merge: a new scored corpus (analysis/quality.py) from an old one. Every row whose prompt every model answered keeps
its prompt fields (request_id, prompt, input_len, dataset, source, is_harmful, reference_text); each model's
response becomes the new text and output_length its output token count, and every field derived from the old
text (its judge scores, similarity and reference scores, compression ratio, truncation flag) is dropped: the judge
scores the new text. Rows no answer covers are left out and counted. A failed answer, an answer for a prompt or a
model the corpus lacks, a prompt answered twice by one model, or a duplicate corpus prompt fails the merge.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Sequence

from bench.records import Sampling
from bench.schema import read_jsonl, write_new_jsonl
from routebalance.backend import BackendClient
from routebalance.config import ConfigError
from routebalance.types import Instance
from tools import direct
from tools.campaign import load_plan

PROMPT_FIELDS = ("request_id", "prompt", "input_len", "dataset", "source", "is_harmful", "reference_text")


def by_model(fleet: Sequence[Instance]) -> dict[str, list[Instance]]:
    """{model: its instances in fleet order}."""
    out: dict[str, list[Instance]] = {}
    for inst in fleet:
        out.setdefault(inst.model, []).append(inst)            # default-ok: grouping into a new list
    return out


async def collect(plan: dict, fleet: Sequence[Instance], prompts: Sequence[direct.Prompt], concurrency: int,
                  out: Path) -> tuple[int, int]:
    """(lines written, failed requests)."""
    if concurrency < 1:
        raise ConfigError("concurrency must be at least 1")
    sampling = Sampling.model_validate(plan["bench"]["sampling"])
    run_id = uuid.uuid4().hex[:8]
    work = [(inst, [(i, p) for i, p in enumerate(prompts) if i % len(insts) == k])
            for insts in by_model(fleet).values() for k, inst in enumerate(insts)]
    written = failed = 0
    async with direct.session(concurrency * len(fleet) + len(fleet)) as sess:
        backend = BackendClient(sess, plan["bench"]["request_timeout_s"])
        await direct.drain(sess, fleet, timeout_s=plan["drain"]["timeout_s"])
        w = plan["warmup"]
        await direct.warm_up(backend, fleet, requests_per_instance=w["requests_per_instance"], max_tokens=w["max_tokens"],
                             prompt=w["prompt"], sampling=sampling, run_id=run_id)

        async def one(inst: Instance, mine: list) -> list[dict]:
            run = await direct.closed_loop(backend, inst, [p for _, p in mine], concurrency, sampling,
                                           request_prefix=f"collect.{run_id}.{inst.instance_id}", fillers=[])
            return [direct.outcome(res, prompt_id=p.prompt_id, prompt=p.text, model=inst.model,
                                   instance_id=inst.instance_id, row=i)
                    for (i, p), res in zip(mine, run.measured, strict=True)]

        runs = await asyncio.gather(*(one(inst, mine) for inst, mine in work), return_exceptions=True)
        lines = [line for r in runs if not isinstance(r, BaseException) for line in r]
        write_new_jsonl(out, lines)
        written, failed = len(lines), sum(not x["success"] for x in lines)
        errors = [r for r in runs if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        await direct.drain(sess, fleet, timeout_s=plan["drain"]["timeout_s"])
    return written, failed


def merge(corpus: Path, answers: Path, out: Path) -> tuple[int, int]:
    """(rows written, corpus rows left out because not every model answered them)."""
    rows: dict[str, dict] = {}
    for lineno, rec in read_jsonl(corpus):
        if rec["prompt"] in rows:
            raise ConfigError(f"{corpus}:{lineno}: a duplicate prompt")
        rows[rec["prompt"]] = rec
    new: dict[str, dict[str, dict]] = {}
    for lineno, a in read_jsonl(answers):
        where = f"{answers}:{lineno}"
        if not a["success"]:
            raise ConfigError(f"{where}: a failed answer ({a['error']})")
        if a["prompt"] not in rows:
            raise ConfigError(f"{where}: prompt {a['prompt_id']} is not in {corpus}")
        if a["model"] not in rows[a["prompt"]]["models"]:
            raise ConfigError(f"{where}: {a['model']} is not a model of {corpus}")
        per = new.setdefault(a["prompt"], {})                   # default-ok: grouping into a new dict
        if a["model"] in per:
            raise ConfigError(f"{where}: {a['model']} answered prompt {a['prompt_id']} twice")
        per[a["model"]] = a
    lines, left_out = [], 0
    for prompt, rec in rows.items():
        if prompt not in new or set(new[prompt]) != set(rec["models"]):
            left_out += 1                                       # not every model answered this prompt
            continue
        got = new[prompt]
        lines.append({**{k: rec[k] for k in PROMPT_FIELDS},
                      "models": {m: {"response": got[m]["text"], "output_length": got[m]["output_tokens"],
                                     "llm_judge_scores": {}}
                                 for m in rec["models"]}})
    write_new_jsonl(out, lines)
    return len(lines), left_out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    for name in ("render", "fleet", "hosts", "data-root", "out"):
        c.add_argument(f"--{name}", type=Path, required=True)
    c.add_argument("--fleet-name", required=True)
    c.add_argument("--dataset", required=True)
    c.add_argument("--concurrency", type=int, required=True)
    m = sub.add_parser("merge")
    for name in ("corpus", "answers", "out"):
        m.add_argument(f"--{name}", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise ConfigError(f"{a.out} exists; the output is always a new file")
    if a.cmd == "collect":
        plan = load_plan(a.render)
        fleet = direct.lease_fleet(plan, a.fleet_name, a.fleet, a.hosts)
        prompts = direct.lease_prompts(plan, a.dataset, a.data_root)
        written, failed = asyncio.run(collect(plan, fleet, prompts, a.concurrency, a.out))
        print(f"{a.out}: {written} answers ({len(prompts)} prompts x {len(by_model(fleet))} models), {failed} failed")
        return 0 if failed == 0 and written == len(prompts) * len(by_model(fleet)) else 1
    written, left_out = merge(a.corpus, a.answers, a.out)
    print(f"{a.out}: {written} rows; {left_out} corpus rows left out (not every model answered them)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
