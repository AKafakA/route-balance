# Static gate — triage of every finding on the live tree (2026-08-01)

The gate went from 18 findings to **4** baselined ones (re-verified 2026-08-01: the gate prints `baseline: 4 known, 4 current`, and `static_gate_baseline.json` holds 4 entries — `llm_judge_scorer.py:581` and `:991` are separate findings, counted as one in an earlier draft that said 3). **14 of the 15 removed were the gate
being wrong, not the code being fixed.** That matters: a gate with a high false-positive
rate gets ignored within a day, which is the same failure as having no gate.

## Detector corrections (the gate was wrong)

| finding | why it was a false positive |
|---|---|
| `cara_serve.py:3055`, `:3149` "host→GPU map defined here" | Both lines are `_HOST_GPU = _host_gpu_map()` — the **D26 fix**, two loaders resolving through one helper. `HOSTMAP_RE` matched any assignment; it now matches only a **dict literal**, and only fails when **more than one** definition exists. One canonical map at `cara_serve.py:180` is the healthy state and is now reported as such. |
| 5 × `offline_training/*` model ladders | Training and data-prep scripts. A wrong ladder there fails loudly at train time (no data for the tier); it cannot mis-route a request. Allowlisted with that reason. Verified: nothing under `offline_training/` is imported by `cara_serve`. |
| 2 × `results = [None] * len(pairs)` | `None` is the canonical "not filled in yet" accumulator — it cannot be mistaken for a measurement and every consumer must test it. `[False] * n` is different in kind: `False` **is** a valid safety label, so it flows on as data. `None` removed from `SYNTH_RE`. |
| `cara_serve.py:473` `rso.get('budget_cost', request_json.get('budget_cost', 0.0)) or 0.0` | Every default in the chain is the same falsy constant as the fallback, so no reachable value is ever replaced. `_benign` now walks nested `.get()` chains instead of comparing only the outermost default. |
| class 4 `batch_scoring` "dead key" | The key appears in **no config and no code** — the gate invented its own candidate list and then reported the invention. Candidates are now derived from the config JSONs on disk. A detector that supplies its own evidence is exactly the failure this file exists to stop. |

## Baselined — real, reported, awaiting an author decision

`static_gate_baseline.json` holds these three. They do not block; **anything new fails.**

1. **`llm_judge_scorer.py:581` and `:991` — `is_harmful = [False] * len(pairs)`.**
   Real (defect G2): a corpus that merely omits the field silently selects the
   **non-safety** rubric. The code already carries a `v3 FIX (G2)` comment and logs a
   loud warning, and the default was **deliberately kept so existing callers do not
   break**. So this is an author-decided state, not an oversight — but it is the last
   place a fabricated value still reaches a scoring path.
   **Owed:** decide whether to make it raise. It affects which rubric scored the
   training corpus, i.e. the quality labels everything else depends on.

2. **`cara_serve.py:2010` (was cited as `:1956`; re-verified 2026-08-01) — a hardcoded Qwen ladder.**
   Inside `if os.environ.get("CARA_DUMMY_ESTIMATOR") == "1":`, the bottleneck-isolation
   harness that must never be on for a cell. Low risk, but it lives in the serving file,
   so `cara_serve.py` stays scanned for class 2 rather than being allowlisted wholesale.

3. **`plot_predictor_comparison.py:83` — an `or`-chain on `best_model_accuracy`.**
   A plotting script, so it cannot affect a served decision, but BestAcc is a reported
   metric and a legitimate 0.0 would fall through. Cheap to fix; not on the S2 path.

## How the baseline is enforced

```
static_defect_gate.py --baseline v3_scripts/static_gate_baseline.json   # exit 2 on anything NEW
static_defect_gate.py --update-baseline                                 # only after triage
```

`gate_selftest.py` asserts all three behaviours: the live tree matches the baseline; an
injected class-1/2/3 defect fails **both** bare and against the baseline (a baseline that
swallowed new findings would be worse than no gate); and a **missing** baseline file fails
rather than passing silently.
