#!/usr/bin/env python3
"""The pairs the main scorer left unscored, judged again through a fixed escalation ladder until every pair is scored.

  stage 1  native retry   re-run the missing pairs through DeepEval's normal path (its own retries apply inside
                          each call)
  stage 2  sync re-run    re-run them synchronously, N attempts each, with the same greedy decoding
  stage 3  temperature    last resort: the same prompt, criteria, evaluation steps and logprob-weighted score, only
                          the sampling temperature changes; each pair filled here is tagged `_judge_temperature`

Stages 1 and 2 never change the judging configuration, and the score is always computed by DeepEval's G-Eval. The
judge reads the prompt, the answer and the reference cut to 2,048 characters, as the main scorer does
(tools/geval_score.py). The corpus is rewritten in place: run it on a copy of judge_assemble.sh's output.

    python -m tools.judge_gapfill CORPUS.jsonl --base-url URL [--judge-key K] [--attempts 8]
                                  [--temps 0.2 0.4 0.7 1.0] [--stop-after {1,2,3}]
"""
import argparse
import json
import sys

NAME = "Quality"
CRITERIA = ("Rate how well the actual output answers the question compared to the expected "
            "output. Consider correctness, completeness, and helpfulness.")


def missing(rows, key):
    return [(i, m) for i, r in enumerate(rows)
            for m, md in (r.get("models") or {}).items()
            if key not in ((md or {}).get("llm_judge_scores") or {})]


def build_metric(base_url, model, temperature=None):
    from deepeval.models import LocalModel
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    kw = {} if temperature is None else {"temperature": temperature}
    lm = LocalModel(model=model, base_url=base_url, api_key="x", **kw)
    return GEval(name=NAME, criteria=CRITERIA,
                 evaluation_params=[LLMTestCaseParams.INPUT,
                                    LLMTestCaseParams.ACTUAL_OUTPUT,
                                    LLMTestCaseParams.EXPECTED_OUTPUT],
                 model=lm, async_mode=False)


def try_pairs(rows, pairs, metric, key, attempts, temperature=None):
    from deepeval.test_case import LLMTestCase
    got = 0
    for i, m in pairs:
        r = rows[i]
        md = r["models"][m]
        tc = LLMTestCase(input=(r.get("prompt") or "")[:2048],   # the main scorer's cut (tools/geval_score.py)
                         actual_output=(md.get("response") or "")[:2048],
                         expected_output=(r.get("reference_text") or "")[:2048])
        for _ in range(attempts):
            try:
                metric.measure(tc)
                if metric.score is not None:
                    md.setdefault("llm_judge_scores", {})[key] = float(metric.score)
                    if temperature is not None:
                        md["_judge_temperature"] = temperature
                    got += 1
                    break
            except Exception:
                continue
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--judge-key", default="deepeval-llama3.1-8b-it_reference")
    ap.add_argument("--attempts", type=int, default=8)
    ap.add_argument("--temps", type=float, nargs="*", default=[0.2, 0.4, 0.7, 1.0])
    ap.add_argument("--stop-after", type=int, default=3, choices=[1, 2, 3])
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.corpus) if l.strip()]
    total = sum(len(r.get("models") or {}) for r in rows)
    miss = missing(rows, a.judge_key)
    print("start: %d/%d pairs missing (%.4f%% coverage)"
          % (len(miss), total, 100.0 * (total - len(miss)) / max(total, 1)))
    if not miss:
        print("already complete — no escalation needed")
        return 0

    # stage 1+2: default greedy decoding, no config change
    for stage, att in ((1, 1), (2, a.attempts)):
        if a.stop_after < stage or not miss:
            break
        metric = build_metric(a.base_url, a.model, temperature=None)
        got = try_pairs(rows, miss, metric, a.judge_key, att)
        miss = missing(rows, a.judge_key)
        print("stage %d (%s, %d attempt(s)): recovered %d | %d still missing"
              % (stage, "native re-run" if stage == 1 else "sync re-run", att, got, len(miss)))

    # stage 3: LAST RESORT — sampling temperature only
    if miss and a.stop_after >= 3:
        print("stage 3 (TEMPERATURE BACKUP — only after 1 and 2 failed):")
        for T in a.temps:
            if not miss:
                break
            metric = build_metric(a.base_url, a.model, temperature=T)
            got = try_pairs(rows, miss, metric, a.judge_key, 1, temperature=T)
            miss = missing(rows, a.judge_key)
            print("   temperature %.1f: recovered %d | %d still missing" % (T, got, len(miss)))

    with open(a.corpus, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tagged = sum(1 for r in rows for md in r["models"].values() if "_judge_temperature" in md)
    print("final: %d/%d missing | %d pair(s) required the temperature backup"
          % (len(miss), total, tagged))
    return 0 if not miss else 1


if __name__ == "__main__":
    sys.exit(main())
