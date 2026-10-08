# v3 DEFECT INVENTORY — every known issue, consolidated

Built 2026-07-30 because the defect list was scattered across four places and no single file held
it: `POST_REBUTTAL_CR_NOTES.md` (CR-1…CR-8), `V2_RERUN_PLAN_2026-06-07.md` §1–2, the session task
list, and `2026-07-25_window_execution_log.md`. **There is only one CR-notes file** — verified by
search; anything remembered as "in the CR notes" but absent below needs a pointer.

Column **v3?** = does this change a number in a v3 rerun.
Status **VERIFIED** = I opened the file/cell this session. **RECORDED** = taken from a handoff
that cites its own evidence, not re-verified here.

---

## 🚨 §00. THE REVIEWED ARTIFACT — read before touching any paper claim

**`Block/cara_paper/cara_neurips.tex` is NOT the submitted paper.** Verified 2026-07-30 by
number-diffing against the OpenReview PDF (`~/nips.pdf`):

| marker | reviewed PDF | `cara_neurips.tex` |
|---|---:|---:|
| "243" (configurations) | 6 | **0** |
| "860" (K requests) | 1 | **0** |
| "203" (routing cells) | 3 | **0** |
| "760" (K requests) | **0** | 3 |
| "215" (cells) | 3 | 6 |

`cara_neurips.tex` is an **earlier draft** — 215 cells / 760K requests, against the submission's
243 configurations / 860K requests / 203 routing cells.

**Authoritative submitted source: `~/Code/llm/route-balance-paper/neurips_2026.tex`**, md5-verified
equal to the OpenReview submission zip (commit `1379ecb`, *"archive: exact NeurIPS submission zip
from OpenReview (md5-verified == neurips_2026.tex)"*). Submitted state = `3226c1a`.

Also present in that repo and **NOT** the reviewed artifact: `neurips_2026_v2.tex` (the Jun-13 v2
rewrite, never submitted) and `eurosys_2027/main.tex`.

This is a **repeat trap** — `feedback_identify_reviewed_artifact_first` records a prior incident
where three findings and a whole reviewer simulation were verified against the wrong paper. It
nearly recurred today: `cara_neurips.tex` contains a *"5× lower mean TTFT (51 ms vs 268 ms)"*
cross-system claim and a whole §5.2 observation (iii) on TTFT. **None of that is in the submitted
paper.** Anchor every paper-facing claim to `neurips_2026.tex`.

---

## §0. Partition by artifact — what actually belongs in this CR

Target = **fix v1 + the rebuttal → the CR of the NeurIPS paper.** A transcript sweep of all 7
sessions (191 MB) produced 863 candidate statements, 105 citing a concrete mechanism. Most do
**not** belong here. The cut that matters:

| class | belongs in this CR? | examples |
|---|---|---|
| **① Affects the SUBMITTED v1 paper** | **YES — CR-critical** | D12′ (Fig 2 BR points), D2 (coverage), D1 (routing signal), D3 (3B suppression), D10 (k), D5 (cap), D14 (public repro script), D15 (Appendix B), D16 (TTFT conflict) |
| **② Affects the REBUTTAL cells** | **YES — the rebuttal goes into the CR** | Port #4 Gemma tiers, Gemma CI inflation, `arm4_recal.sh`, dead `vllm_params.port`, AvengersPro flood cells, static-prior λ6 back-check |
| **③ v2-only** | **NO — v2 was never submitted** | batching-ablation POST-format bug (`cara_serve.py:634`), `compute_ci_and_tail.py:58` latent or-bug (May path, feeds no v2 number), `e10_microbench.py` map path, `neurips_2026_v2.tex:421` λ=6 footnote, budget-filter pipeline no-op |
| **④ My own reproduction / tooling errors** | **NO — not paper defects** | rebuilding the KNN at k=50, missing `CARA_INPROC_PREDICTOR=1`, smoke-tally bash subscript, `transform_block…` dir rename, `setup_p100_smoke` transformers re-pin |

**Class ④ is a real share of the 863 and must not be mistaken for paper defects.** Recorded
verbatim in-session: *"The k=50 thing was MY reproduction bug, not the submission's"* — the
submission served k=10; I rebuilt at k=50 and produced a flood. Likewise the 3B-concentration
"bug" traced to a missing `CARA_INPROC_PREDICTOR=1` in **my** deployment, not the paper's.

Only ① and ② are v3 scope. The table below is filtered to those.

### Class ② — rebuttal-cell defects (all already fixed or dispositioned; listed so v3 does not reintroduce them)

| item | what happened | state |
|---|---|---|
| Port #4 | `cara_serve.py:1500-1506` tier ladder hardcoded to Qwen; `"3b"` never matches `google/gemma-3-1b-it` → `CARA_STATIC_LAT` silently ignored | caught pre-run, fixed, re-verified |
| Gemma CI inflation ~1.4× | local `gemma3_scored_full.jsonl` is a **Qwen-keyed** file; the real judged corpus `gemma3_merged_test.jsonl` was never synced from node0 | root-caused; **confirm the posted Gemma CIs used the correct file** |
| `arm4_recal.sh` | served the preset but never set `AUDIT_STATIC_LAT`, so `cell_audit.sh` compared against the original and failed all 12 cells — then the script touched the done-marker anyway and **reported "complete"** with 0 cells produced | fixed; a false-success pattern to gate against |
| `vllm_params.port` | dead config — no `--port` ever reaches the cmdline; backends run on vLLM's default 8000. Generated jul25 configs inherited a dead 7000, making healthy backends look dead | fixed |
| Gemma `model_deployment.json` | carried `hf_model_name: google/gemma-3-1b-**pt**` (base, not instruction-tuned) | fixed — **verify the served ladder was `-it` throughout** |
| AvengersPro flood | 36 cells invalid-as-AP → rerun | rerun, originals deleted |
| static-prior back-check | λ6–12: 3B exact (12.2–12.3), **72B off by 2.2 ms/tok at λ6**, 7B/14B within 2–3 | dispositioned |

---

## A. Defects that change numbers — v3 MUST fix

| # | defect | evidence | v3 action | status |
|---|---|---|---|---|
| **D1** | **Routing signal ≠ reported metric.** Submitted routes on `score_type=judge` (Qwen2.5-7B-Instruct) while every reported quality figure is deepeval. r = 0.22 over 59,636 paired scores. | `V2_RERUN_PLAN:20`; drift audit §1 | `score_type` → **deepeval** | RECORDED |
| **D2-verdict** | ✅ **A4 RE-AGGREGATION VERDICT (measured 2026-07-30): no posted CLAIM is invalidated; one numeric bound tightens.** Recomputed POST 2's arms both ways over λ{6,8,10,12,18,24}. **(i) Chain validated** — the buggy basis reproduces every posted value to 4 dp (λ6 0.5074/0.5096, λ8 0.5076/0.5074, λ10 0.5073/0.5084, λ12 0.5073/0.5095, λ18 0.5060/0.5079, λ24 0.5043/0.5080). **(ii) Coverage** 75.9–76.6% → **100.0%** zero-safe, as CR-1 predicted. **(iii) Quality** falls ~0.120 uniformly (0.507→0.387), consistent with the recorded +0.116. **(iv) The claim holds** — POST 2 asserts quality *indistinguishable* and explicitly claims **no** quality advantage for the learned arm. On the zero-safe basis every paired Δq is **negative** (learned ≤ static), so the concession is if anything more clearly correct. **(v) One bound moves**: POST 2's "every \|Δq\| ≤ 0.004" becomes **≤ 0.005** zero-safe (λ12 0.0047, λ18 0.0045). **(vi) One sign flip at λ8** (+0.0002 → −0.0021) — the buggy value was already ~0, so this is noise-level, not a reversal. **(vii) The 8.1–9.8% cost claim is untouched** — a cost ratio cannot move with a quality bug. | measured 2026-07-30, `scratchpad/a4_reagg.py` over `v1exact_bridge/v1x_arm1_uniform_*` and `v1exact_recal/v1x_recal_uniform_*` | CR: restate the bound as ≤0.005 alongside the zero-safe numbers | **VERIFIED — CLOSED** |
| **D2** | **Aggregator drops 0.0 scores.** `per_model.get(m) or per_model.get(stripped)` — a deepeval 0.0 falls through to a never-stored key. **This IS the paper's 72–78% "coverage"**, misdescribed in the submitted appendix as unscored lookup pairs. +0.116 inflation (0.512 → 0.397). | `aggregate_full_215.py:185`, `aggregate_overnight_grid.py:129`; CR-1 | zero-safe aggregation everywhere | RECORDED |
| **D3** | **A30 latency tier first-match-P100 bug.** Live measured effect is **large**, not cosmetic: fixing it moved the arm-1 λ6 mix to 3B 74.8% / 14B 21.1% / 7B 4.1% / 72B 0.1%, against May's 14B-heavy profile. **Mechanism confirmed by static audit (A5, 2026-07-30):** the instance→tier map matches on `model_name` **alone** and `break`s on first match — `for inst_type, meta in inst_metadata.items(): if meta.get("model_name") == inst._model_name: … break`. With both `qwen2.5-3b_p100` and `qwen2.5-3b_a30` declaring `model_name: Qwen/Qwen2.5-3B`, JSON insertion order decides, so A30-hosted 3B instances inherit the P100 latency tier. **D3 and D4 are one defect with two fixes** — match on (model, GPU) in code; drop the P100 entries in config. Do both. | ⚠️ **citation corrected**: the recorded `cara_serve.py:2650` does **not** resolve — `:2650` in the v1exact tree is the model-estimator factory. Real site: **`cara_serve.py:2662-2668`** (v1exact tree) | **v3 ports the v2 fix — do not re-implement.** v2 fixed this at `_load_scheduler_config` (current tree `:2753-2760`), deriving GPU from the host prefix: `_HOST_GPU = {"d7525":"a30","c240g5":"p100","c4130":"v100","d8545":"a100"}`, with a comment naming the bug. **But the fix landed in only one of two paths** — `_load_learned_predictors` (`:2838-2841`) still matches on `model_name` alone. The two are `if/elif` (`:3027-3031`), and every v1exact/Gemma cell passes the md5-gated `--scheduler-config`, so the fixed path is what ran. The legacy copy is **dormant but latent**: any launch omitting that flag silently gets the old behaviour. Fix both in v3. | **VERIFIED (static)** |
| **D4** | **The zombie P100 tier is DEPLOYED, not merely configured** (A3, 2026-07-30). `models_full/cara/latency/deploy/` ships **`qwen2.5-3b_p100.xgb`** beside `qwen2.5-3b_a30.xgb`, and `xgboost_metadata.json:instance_types` lists `qwen2.5-3b_p100` among its five entries — that list is what the tier matcher iterates, which is what makes D3's first-match ambiguity possible at all. No P100 exists in the leased topology (c4130 = V100, d7525 = A30, d8545 = A100). **Author 2026-07-30: remove the zombie head.** Scope is wider than first recorded: the **artifact**, its **metadata entry**, and the **config capacities** — all three. | verified 2026-07-30: `latency/deploy/` listing + `xgboost_metadata.json`; host sweep | remove artifact + metadata entry + config entries | **VERIFIED** |
| **D23** | **The paper's own split arithmetic does not close.** `neurips_2026.tex:219` states a dataset of **18,608** prompts "split 80/20 into **14,919** train and **3,634** test". But 14,919 + 3,634 = **18,553**, a **55-prompt gap**; a true 80/20 of 18,608 would be 14,886 / 3,722, matching neither. **Four train counts now exist in the record**, all legitimate at different pipeline stages: **14,966** (served index, pre-filter — D24), **14,919** (published, post-filter), **14,915** (`train_flat`, used to pin the Gemma corpus — "4-row delta vs published, immaterial"), and the implied 14,886 (a literal 80/20). The defect is that the paper reports one of them beside a total that belongs to another stage. | verified 2026-07-30: arithmetic on `:219`; `2026-07-25_window_execution_log.md:144`, `:326` | at CR, state the filter stage each count belongs to and make total = train + test at the reported stage | **VERIFIED — NEW** |
| **D24** | ✅ **RESOLVED — benign, NO LEAKAGE (author-directed trace, 2026-07-30).** The served index's 14,966 rows are the **pre-filter** build: 14,919 filtered rows plus **47 rows the short-response filter (`--min-output-tokens`, `prepare_benchmark_data.py:1166`) would later drop**. They are train-side rows, not test prompts. The same verification line confirms the domain masks matched **14,919/14,919 exactly**. No train/test contamination; v1 quality numbers are not affected on this axis. | verified 2026-07-30: `2026-07-25_window_execution_log.md:326` — *"(14,919/14,919 matched, mask sizes == domain counts EXACTLY, 47 pre-filter rows kept)"* | none — record the provenance in the CR so the two counts are not read as a discrepancy | **RESOLVED — CLOSED** |
| **D5** | **Cap mismatch: corpus 1024 vs serving 256.** Corpus max `output_length` = 1023 over 32,344 pairs, 0 truncated. Serving: Qwen `v1x_ap_po2_l12` max 256 with **841/3,534 = 23.8%** at cap; Gemma `gemma_rb_uniform_l12` max 256 with **610/1,979 = 30.8%**. The KNN length estimator is fitted on a distribution the server cannot produce and over-predicts most for verbose large tiers; predicted length feeds **both** the analytic e2e latency term and the cost term. | measured 2026-07-30 from the artifacts | **fix the mismatch** — direction is an author call (raise serving → 1024, or rebuild corpus at 256) | **VERIFIED — NEW** |
| **D6** | **Param grids silently ≠ May.** cara = 8 simplex tuples (not scalar-q); AvengersPro = pw{.25,.39,.53,.70,.80} (not lambda_cost{1,3,10,30}); BEST-Route = argmax + t{.3,.5,.7}; dispatcher-only adds **random**. Paper §5.2 names AP pw{.25,.39,.53,.8}, BR t{.3,.5,.7}, passthru{rr,sq,ra}. | `V2_RERUN_PLAN:§2.1` | grid must match the paper's §5.2 names exactly | RECORDED |
| **D7** | **AvengersPro cost vector is unrecoverable.** The paper's Qwen AP was *not* built on the size proxy (measured: pw0.25 vs pw0.7 agree in only 2/32 clusters; 72B tops 8/32, which a 24× size penalty forbids). **But** the artifact records pw/cs and the formula and **not the cost vector**, and `train_scored_filtered.jsonl` carries no cost field — so Qwen's exact prices cannot be reconstructed. | exec log 2026-07-28T12:47Z; task #11 | rebuild Qwen AP with **`--price-per-token` explicitly passed** so prices are recorded | RECORDED |
| **D8** | **AvengersPro price-proxy trap on any new family.** The repo builder's default cost term is **model size**, which assumes price tracks size — true on Qwen (0.06→0.40 monotone), **false on Gemma** (1B dearest at 0.100/1M, 4B cheapest at 0.040). On the size proxy AP would rank 1B cheapest — the exact error a family experiment exists to expose. Fixed for Gemma by injecting real prices. | exec log 2026-07-28T12:45Z | never use the size proxy; pass prices for every family | RECORDED |
| **D9** | **AvengersPro all-72B flood.** Upstream `SimpleClusterRouter` interface drift (`embedder.get` + `tokenizer`). Shim fixed — but the live flood persisted because `cara_serve` had the stale module cached; needed a restart. 20 `FALLBACK_INVALID` degenerate cells were produced and later deleted. | `V2_RERUN_PLAN:§2.2`; checkpoint §4 | shim + **mandatory restart** after any router change; degenerate-mix halt per cell | RECORDED |
| **D10** | **KNN k: paper says k=50 / 14,919; the served v1 artifact is the Mar-15 pickle with k=10 baked in over 14,966 rows.** Configured k=50 was dead for `type="default"`. | CR-3; `project_v1_knn_true_artifact` | v3 keeps **k=50** per the author's keep-unless-broken rule — but the *artifact* must actually honour it, which the v1 one did not | RECORDED |
| **D11** | **Cost axis convention.** The rebuttal prices **input+output**; the submission prices **output only**. Absolute $/1k do not line up between them. | CR-2 | pick one convention, state it, apply to every v3 cell | RECORDED |
| **D12** | ⚠️ **CORRECTED 2026-07-30 — see D12′ below.** My earlier action ("run t=0.5 as the headline") was wrong: t=0.5 is the bug-contaminated point, not the trustworthy one. | — | superseded | **WITHDRAWN** |
| **D12′** | **BEST-Route t0.5/t0.7 fallback bug — the largest single finding.** When `top_prob < 0.5` the code falls back to `model_pool[0]`, which is **14B**, while the docstring says "smallest" (3B). At t0.5 **98.4%** of traffic is fallback-dominated; at t0.7, **99.9%**. So the gated traffic landed on an accidental model. Consequence recorded verbatim in-session: *"Submitted v1 paper — Figure-2 t0.5/t0.7 points invalid as labeled; text:241 contradicted by own data; argmax rows, AP, PT, RB all clean."* Figure 2's BEST-Route frontier is one real point (t0.3, near-argmax) plus **two bug artifacts**. The intended t0.5 point would have been ~98% → 3B, a cheap low-quality cascade point — i.e. the baseline's "strong quality at t0.5" is the bug being *lucky*, at real latency/cost expense. | transcript trace 2026-07-30; `best_route_4way.py` | fix the fallback to honour the docstring, then re-run the **full t-sweep {0.3,0.5,0.7}**; reframe Fig 2; rewrite text:241 | **VERIFIED (in-session record)** |
| **D14** | **Public reproduction script carries the zero-drop bug.** `reproduce/aggregate_results.py` in the **published** `route-balance` repo uses `score = get(m) or get(stripped)` — drops **855 valid 0.0 DeepEval scores** (coverage 0.758), inflating uniform quality **0.371 → 0.489**. Anyone reproducing from the public artifact gets the inflated numbers. | transcript trace; public repo | fix in the public repo (author-gated push, task #5) | **VERIFIED (in-session record)** |
| **D15** | ❌ **VOID — mis-anchored to the stale draft (A5, 2026-07-30).** Recorded as "Appendix B documents a linear schedule; the code is three-step." **The submitted paper has no batching appendix** — it has exactly two: bootstrap CI (`sec:appendix-bootstrap`) and scheduling-overhead decomposition (`sec:appendix-overhead`). The `\section{Adaptive batching schedule}` exists only in `cara_neurips.tex`, the quarantined draft. The submitted paper describes the limit **qualitatively** at `:136` — *"the limit grows when more instances are busy and shrinks when the cluster is mostly idle"* — which the code's three-step schedule (`ratio<0.3→4/50 ms`, `<0.7→16/100 ms`, `else 32/200 ms`, `cara_serve.py:228-240`) satisfies exactly. **Nothing owed. POST 1's formula quote to reviewer 287j was correct.** | verified 2026-07-30: `neurips_2026.tex:136`, `:316`, `:337`; `cara_serve.py:228-240` | none — remove from the CR list | **VOID** |
| **D20** | ✅ **E2E formation verified correct — no defect.** POST 2 told gTfM that end-to-end latency is formed *analytically*, not learned. Confirmed in the submitted code at `cara_serve.py:1035-1043`: `slot_open → own_out*tpot`; else `queue_iters = pending_dec/max(running_n,1)` and `(queue_iters+own_out)*tpot`. That matches `neurips_2026.tex:207` term for term, including "if the instance has a free decode slot, only the second term applies". The `ttft + expected_length*tpot` at `:1116` is the **defensive fallback** taken only when `sidecar_e2e is None`. **Paper, POST 2 and code all agree.** | verified 2026-07-30 | none. Minor doc nit: the comment at `:1113` says "Prefer dedicated E2E XGBoost prediction", which is stale — E2E is analytic, not an XGBoost head | **VERIFIED — CLOSED** |
| **D16** | ✅ **RESOLVED 2026-07-30 — no posted statement is false; CR-4 stands.** The TTFT gate at `cara_serve.py:1360-1371` is real, but the entire filter block is guarded by `_filter_on_check = slo_defaults.get("filter_enabled", True)`, and **every deployed config sets `filter_enabled=False`** — verified by opening them: `_cluster_archive_20260506/scheduler_config_carafast.json` (submitted) and `scheduler_config_gemma.json` both `False`. With the filter off, TTFT is not even predicted: `:1010` requests only `["e2e"]`, and `:1030-1031` sets `ttft = 0.05` as a placeholder with the comment *"not used downstream when filter off."* So TTFT genuinely is **not** a term in any reported run. The earlier in-session claim that "our rebuttal wording is wrong" read the gate without checking its guard. **POST 2's statement to gTfM is correct. CR-4 is correct.** | verified 2026-07-30: `cara_serve.py:1010,1030-1031,1350-1371` + both deployed configs | CR nuance only: Algorithm 1 may correctly describe a TTFT gate that `filter_enabled=false` disables in every reported run — say so, rather than removing it | **VERIFIED — CLOSED** |
| **D17** | ⚠️ **RE-CLASSIFIED 2026-07-30 — not dead weight, it is the event-loop offload switch.** `batch_scoring` (`best_route_4way.py:32,43,121`) selects between two scoring paths. When **False** (the default, and set by nothing) `_route_one` runs the tokenizer and `self._model(**enc)` forward **inline in the async coroutine** (`:163-173`), blocking the scheduler event loop for every request. When **True**, `_score_batch_sync` is dispatched via `loop.run_in_executor` (`:110`, `:123`). **This is the actual root cause of the BEST-Route collapse** — the ~18.73 req/s ceiling is the DeBERTa scoring rate, independent of which model the router picks, so it survives the D12′ fallback fix (author, from testing). | verified 2026-07-30: `best_route_4way.py:110,123,163-173` | **Keep `False` in v3.** Enabling it re-architects a baseline — that is the EuroSys "enhanced baseline" variant, out of NeurIPS CR scope. §5.3 correctly describes the serial-scoring behaviour as an architectural property of BEST-Route. Document the flag; do not flip it. | **VERIFIED — do not "fix"** |
| **D18** | **Our Po2 is not standard Po2.** `dispatch/block_style.py` implements a **predictive** variant (samples 2, picks by *sidecar-predicted latency*), which is not what reviewer faAU asked for. | transcript trace | describe it accurately, or run standard Po2 alongside | RECORDED |
| **D13** | **`cara_serve.py` tier matching is a hardcoded Qwen ladder** (`"72b","14b","7b","3b"`). `"3b"` never matches `google/gemma-3-1b-it`, so `CARA_STATIC_LAT` was silently ignored for Gemma — no error. Fixed via Port #4 for the Gemma basis only. **Confirmed by static audit (A5, 2026-07-30):** it is the env-gated arm-4 port — `CARA_STATIC_LAT` unset ⇒ `_static_tpot` None ⇒ byte-identical v1 behaviour, so it is inert for every non-arm-4 cell. | ⚠️ **citation corrected**: recorded `:1500-1506`; real site is **`cara_serve.py:1467-1478`** in the v1exact(-v2) tree (`_sl = os.environ.get("CARA_STATIC_LAT")` at `:1467`, ladder at `:1478`) | family-gated tier matching in the mainline | **VERIFIED (static)** |
| **D21** | ✅ **Eq. 1 scoring semantics verified — no defect.** The submitted code computes `qual_score = float(pred.get("quality", 0.0))` (raw [0,1]) and `1.0 - (x / (max_x + eps))` for cost, latency and balance, matching the printed Equation 1 and the drift audit's claim. A comment block at `:1457-1460` documents the same. Confirms `quality_norm=raw` is the submitted semantics and is correctly kept unchanged in v3. | verified 2026-07-30: `cara_serve.py:1530-1533`, `:1457-1460` | none | **VERIFIED — CLOSED** |

| **D19** | **The FAISS KNN search never batched — in any version.** `model_estimator.estimate_batch` documents *"Single FAISS/numpy search for all queries"* (`:523`) and the paper frames the estimator as amortizing across the batch (`neurips_2026.tex:84`, §5.3). The embedding **is** batched (`:539`), but the search loops one query at a time: `for i, query in enumerate(query_embeddings)` (`:546`) → `query.reshape(1, -1)` (`:547`) → `search(q32, k)` (`:552`). FAISS `search()` accepts `(N, d)` and vectorizes; this forfeits the claimed amortization. The numpy fallback is per-query too. **All 7 `search()` call sites in the repo pass a single query** — v1exact tree, `_raw/route_balance`, current `block/`, `knn_estimator.py:349`, `may8_knn_estimator.py:349`. The v2 plan listed "batched-FAISS KNN" as an intended fix; **it never landed**, so v2 ran the same loop. | verified 2026-07-30 by reading all call sites | **Implement in v3.** Claim/implementation gap ⇒ defect repair, not a performance improvement, and not asymmetric — it makes RouteBalance match RouteBalance's own stated design. Expect §5.3 decision compute to fall further and the §5.5 bs=1 cliff to steepen; both must be measured, not assumed. | **VERIFIED — NEW** |

| **D25** | **`CELL_MAP` is blind to every Gemma cell.** `build_cell_map.py:49` globs `*_n3534.json` / `*_n3534.json.gz` **only**. Gemma cells are `*_n1979*`, so all 25 are invisible — which is why the map reports `gemma_cells \| 0 cells \| 81 MB` (the MB comes from a directory walk, the count from this glob). **Consequence: no Gemma claim has ever been machine-verified against its cells** — POST 3's W1-B mapping was done by hand. Third instance today of the hardcoded-list bug class, after the Qwen tier ladder (D13) and the P100 tier (D4). | verified 2026-07-30: `build_cell_map.py:49` | glob `*_n*.json*` or parameterize the split size; regenerate CELL_MAP (already required — it was also anchored to the stale draft) | **VERIFIED — NEW** |
| **D26** | **The node→GPU map is duplicated in two files and must stay in sync.** `cara_serve.py:2757` `_HOST_GPU` and `exp/cara/deploy_cara.py:734` `NODE_TO_GPU` both hold `{d7525:a30, c240g5:p100, c4130:v100, d8545:a100}`. **Directly affects the D4 zombie-P100 removal**: dropping P100 from one map and not the other lets a redeploy re-introduce the tier the scheduler was fixed to stop mis-matching. Also breaks silently on any new node type. | verified 2026-07-30 by A8 class-2 sweep | remove P100 from **both**; better, hoist to one shared constant | **VERIFIED — NEW** |
| **D27** | **Falsy-zero on EMA rates in the predictor evaluation script.** `eval_latency_predictors.py:100-101`: `ss.get("ema_prefill_tok_per_s", 1000.0) or 1000.0` and the decode equivalent. A genuine `0.0` rate (idle instance) is falsy and silently becomes the default — the same class as D2. **Not on the serving path**, so no cell is affected; it distorts offline predictor-accuracy evaluation. | verified 2026-07-30 by A8 class-1 sweep | explicit `is None` check | **VERIFIED — NEW (low)** |

## A10 — audit against the REAL posted rebuttal (`OPENREVIEW_AS_POSTED_2026-07-29.md`)

The audit nobody had done: every factual claim in the text reviewers actually read, checked
against code/cells/paper. Started 2026-07-30 after the authoritative file was finally persisted.

| # | posted claim (287j reply) | verdict |
|---|---|---|
| R1 | *"prefix caching was disabled, so prefix-hit effects do not confound the comparison"* | ❌ **FALSE.** APC enabled by default for main-table cells (`deploy_cara.py:154-159`); `--disable-prefix-caching` was a no-op (rebind discards it; all four Qwen models carry top-level `vllm_params`); all 13 live tiers show `prefix_flags=[NONE]`. **Author corrected on OpenReview 2026-07-30 before any reviewer read it.** |
| R2 | adaptive schedule `<0.3→4/50ms`, `<0.7→16/100ms`, else `32/200ms` | ✅ holds — `cara_serve.py:228-240` |
| R3 | *"After each assignment, RB immediately updates its local dead-reckoned view"* | ✅ holds — `:827-841`, called `:1887`/`:2091` |
| R4 | *"TTFT and KV occupancy are not separate objective terms, although KV utilization and free-block count are features of the TPOT predictor"* | ✅ holds — D16/D20 |
| R5 | *"disabling adaptivity increases end-to-end latency by 5.1–6.9%"* | ✅ matches paper §5.5 |
| R6 | *"decision compute falls from 44.7 ms to 25.7 ms"* | ⚠️ **numbers may be under-reported ~2×** — `predict_round_ms` divides a *running* total by `len(batch)` inside the loop, so request *i* reports `(i+1)c/B`. Inherited from v1, so the paper's Appendix Compute column carries it too. Direction unknown until recomputed. |
| R7 | *"Quality, cost, and latency are normalized over the candidates in the same round"* | ⚠️ **ambiguous but defensible.** Code normalizes over **one request's candidate instances** (`max_lat = max(e2es)` where `e2es` is that request's candidates, `:1555-1558`). "the candidates in the same round" can be read that way. **But the PAPER is not defensible** — `neurips_2026.tex:134` says normalized *"against the batch itself"*, which is false. `norm_ctx` (the genuine batch-wide range) is built at `:1843`/`:2035`, updated at `:1859-1860`, and **never read**. Dead code proving the intent was batch-wide and never wired. |

**Owed at CR:** correct `:134` to describe per-request candidate normalization, or wire `norm_ctx`
and make the sentence true. Do **not** repeat "against the batch itself" as written.

### gTfM reply — verified

| claim | verdict |
|---|---|
| 7-rate table (0.5074/0.5096 … 0.5043/0.5080), 8.1–9.8% cheaper, \|Δq\|≤0.004 | ✅ A4 reproduced **cell-exact to 4 dp**; zero-safe basis keeps the claim, bound moves to ≤0.005 |
| static constants 3B 10.7 / 7B 19.9 / 14B 16.5 / 72B 42.6 ms/tok | ✅ matches the recal preset |
| λ24 mix 56%/38% vs 68%/29% | ✅ recomputed 55.6/38.6 vs 67.5/29.3 |
| arm2 load-blind: 69.1% smallest tier, q 0.4899 | ✅ |
| *"head predicts TPOT only … E2E formed analytically, not learned"* | ✅ **verified term-for-term** at `cara_serve.py:1035-1043` (D20) |
| *"decision compute falls 44.7 → 25.7 ms"* | ⚠️ R6 — `predict_round_ms` running-total bug |

### faAU reply — verified, incl. the FIRST cell-level check of the posted Gemma table

`CELL_MAP` was blind to Gemma (D25), so W1-B had never been machine-verified. Recomputed all five
posted rows from `gemma_cells/*_l24_*`:

| system | posted q | zero-safe q | mean_ms posted / cell |
|---|---:|---:|---|
| AvengersPro Po2 | 0.6300 | 0.5631 | 3719 / **3719** |
| RB (0.8,0.1,0.1) | 0.6160 | 0.5499 | 2673 / **2673** |
| RB (0.5,0.3,0.2) | 0.6057 | 0.5386 | 2355 / **2355** |
| RB (1/3,1/3,1/3) | 0.6010 | 0.5256 | 2108 / **2108** |
| Passthrough SQ | 0.5732 | 0.4903 | 1622 / **1622** |

**Latencies match exactly.** Qualities are uniformly higher in the posted table by 0.066–0.083 —
the zero-drop signature, inflating most for systems routing to cheaper tiers. Expected: PATH v4
posted on the submitted basis by design. **Ordering is identical on both bases**, and the posted
"+0.014" AP-vs-RB(0.8,0.1,0.1) gap recomputes to **+0.0132**. ⇒ **no posted Gemma claim is
invalidated.**

Also resolves a discrepancy: the execution log's 13:03Z table showed AP at 0.5803 / mean 7034 with
a degenerate 4b-100% mix. The banked cells give mean **3719**, matching the posted table — so the
AP cells were re-run after the real-price artifact was rebuilt, and the posted table uses the
corrected ones.

| other faAU claim | verdict |
|---|---|
| W3: BR Po2 λ24 = 48182 ms vs RB 2408 (20.0×); collapse upstream of dispatch | consistent with the serial-DeBERTa ceiling (D17) |
| W2-B: index construction 52.3 s | ⚠️ **CR-7 confirmed against the authoritative text** — 52.3 s is the **Qwen** build; Gemma's own was 24 s, inside a Gemma-framed paragraph |
| W2-B: 405k requests / 11.1 h profiling "original-family" | ✅ correctly attributed |

**A10 verdict: no posted claim is invalidated. Two corrections owed at CR** — R1 (prefix caching,
already fixed on OpenReview by the author) and CR-7 (52.3 s family attribution). Two flagged for
recomputation — R6 (decision-compute figures) and the paper's `:134` batch-wide normalization.

## B. Process defects that invalidate cells if repeated

| # | defect | evidence | v3 action |
|---|---|---|---|
| **P1** | **Scheduling-mode phase split.** cara native = `--scheduling cara`; baselines = `--scheduling pipeline`. A bare `scoring_weights` POST in pipeline mode **raises** (`cara_serve.py:474`). Runs must restart between phases. | `V2_RERUN_PLAN:§2.4` | phase-split runner, restart gate between phases |
| **P2** | **Native-mode contract is mandatory** for any `--scheduling cara` cell: `CUDA_VISIBLE_DEVICES=""`, `CARA_INPROC_PREDICTOR=1`, no `OMP_NUM_THREADS=1` — otherwise a silent **600×** predictor slowdown. | v2 MAP note 2 | preflight assert per cell |
| **P3** | **Admission filter is a native-scheduler feature.** In pipeline mode it is a silent no-op — v2 initially measured `cara_filter == cara_nofilter` and had to redo §5.4 in native mode. | exec log 2026-06-08T11:18Z/11:40Z | budget cells native-only, with a filter-cuts smoke gate |
| **P4** | **Latency is not comparable across sessions.** A/B control showed a 5.9% swing at λ28 from run timing alone. Quality reproduced to ±0.0002 across two days. | checkpoint §7 | every latency comparison paired within one session; bracket long runs |

## C. Text / documentation only — no rerun effect

| # | item | source |
|---|---|---|
| T1 | TTFT phrasing implies TTFT is modelled; it is not (decode-dominated score) | CR-4 |
| T2 | Vidur citation promised to gTfM — verify the characterisation before writing | CR-5 |
| T3 | W2-B onboarding paragraph attributes two Qwen figures to Gemma | CR-7 |
| T4 | Cap filter + composition shift must be stated wherever the Gemma table appears | CR-8 |
| T5 | Fig 1 is stale — RoBERTa box never used (deployed = MiniLM+KNN), lists P100, omits Qwen-3B, "Instatnce" typo | rebuttal audit §5 |
| T6 | PAPER_TODO says the v1 cap was 512; it was **256** | drift audit §2 |

## D. Tooling defects — fix, but they do not change v3 numbers

| # | item | source |
|---|---|---|
| G1 | `aggregate_overnight_grid.py` **still carries the zero-drop bug** (only `aggregate_v2_grid.py` was fixed) | CR-1 |
| G2 | `llm_judge_scorer.score_pairs()` silently defaults a missing `is_harmful` to `[False]*len(pairs)` — a corpus omitting the field changes rubric with no warning. Should raise or warn. | CR-6 |
| G3 | `p4_offline_bestacc.py` does not work; closed, will not be fixed, kept labelled | checkpoint §6.4 |
| **G4** | **Stale TTFT code to remove in v3 (author, 2026-07-30).** TTFT was removed from the deployed models — the paper describes **one XGBoost TPOT head** (`neurips_2026.tex:207`) and POST 2 told gTfM the same. Three leftovers reference the removed head: (a) `cara_serve.py:1116` `e2e_latency = ttft + expected_length * tpot` — a fallback branch reachable only when `sidecar_e2e is None`; (b) the comment at `:1113` "Prefer dedicated E2E XGBoost prediction", which is false — E2E is analytic; (c) `:1030-1031` sets `ttft = 0.05` as a placeholder with "not used downstream when filter off". **Not a results defect** — the analytic path at `:1035-1043` is what runs and it matches the paper term for term (D20). Dead-code cleanup only, but it should go, because a reader tracing the latency story hits the stale branch first. |

## D′. NOT DEFECTS — design headroom (author, 2026-07-30)

**Author's framing, and it is the right one:** *"it just means you added a useless feature and
can be improved, not means you are wrong."* These are improvement opportunities. They produce no
wrong number and falsify no claim, so **nothing here is owed as a correction** and none of it
belongs in a change note. Recorded so a future session does not re-file them as defects.

| # | item | why it is not a defect |
|---|---|---|
| **N1** | `xgboost_predictor.py:62` — `decode_ctx_p95 = ss.get(...,1.0) or 1.0`, so `ctx_relative_to_p95` is uninformative on **37.65%** of records (measured; 162× mean shift) | Training and inference share `compute_derived_features`, so the head was **fitted on the same feature it is served** — internally consistent, not a train/serve mismatch. The submitted paper does **not** enumerate, count, or characterise the features (`:199` is qualitative; there is no predictor-accuracy table), and every claim it does make about the head holds: one head per (model,GPU) tier, tier-local QPS sweep, queried in parallel, E2E analytic. MAE figures are honest — measured on held-out data through the same pipeline. **Fixing inference alone would CREATE a mismatch; fixing both needs a refit and reopens D28.** Leave the v3 head alone; improve in the next predictor generation. |
| **N2** | `CaraCDFFilter` receives no calibration data, so it is a deterministic threshold rather than a CDF | `neurips_2026.tex:183` explicitly files the CDF-against-empirical-residual approach as **future work**. The paper never claims it is implemented. |
| **N3** | `batch_scoring` dead flag in the live working copy | Documented in D17; keeping it `False` is the submitted behaviour. Enabling it is the EuroSys "enhanced baseline", out of CR scope. |

**Rule this establishes:** classify by *"does a stated claim become false?"* — not by *"could this
be better?"*. Over-confessing headroom in a change note invites scrutiny of things the paper never
asserted. Cf. `feedback_rebuttal_state_dont_defend`.

## E. Open / unexplained — carried, not fixed

| # | item | disposition |
|---|---|---|
| U1 | The **25% label rung beats the full index** on deepeval (+0.001 on the routing judge). Unexplained; out of the packet. Note D1 is the likely mechanism — if v3 routes on deepeval, this should resolve. **Check it after the v3 rerun.** | out of packet |
| U2 | CR-6 downgraded: safety rubric is not applied to the Gemma corpus. Not a defect — deepeval scored both corpora identically. Optional future enhancement. | not owed |

---

## What v3 keeps (author rule: fix only what is broken)

| v1 setting | keep? | why |
|---|---|---|
| serving cap **256** | **keep** | it is vLLM's own custom-dataset default (`datasets.py:1380`) and the paper's setting — not a defect on its own. Only its *mismatch* with the corpus (D5) is. |
| KNN **k=50** | **keep** | evidenced-minor: R5 band ~0.013 over k∈{5,10,20,50}. It is what the paper describes. v2's k=10 does **not** carry over. |
| `quality_norm` **raw** | **keep** | it is the paper's printed Eq. 1. v2's minmax does **not** carry over. |
| batch config, KV config, relax order, `lpt_sort_key=max`, `budget_tokens=256`, `filter_enabled=false` | **keep** | 48/66 config keys are already identical v1↔v2 |

**Therefore v3 ≠ v2, and v2's corrected headline does not transfer.** RB 0.4184 vs BR 0.4056,
CI [+0.005, +0.022] was measured at k=10 / minmax / cap 1024. v3 must establish its own numbers.

---

## Gaps in this inventory — say so rather than guess

- "Evaluator bugs" plural: I found **D2** (aggregator zero-drop), **G1** (the second aggregator
  still carrying it) and **G2** (silent `is_harmful` default). If another evaluator defect is
  remembered, it is not in any file I searched and needs a pointer.
- Three HF-dataset bugs were fixed 2026-03-23 (`feedback_latency_data_provenance`); they predate
  the submitted run and are assumed already reflected in it. **Not re-verified.**
- D1, D3, D6–D13 are RECORDED, not re-verified this session. The §3a gating audit in
  `2026-07-30_v3_CORRECTED_SWEEP_PLAN.md` must confirm each against code before v3 is built.
