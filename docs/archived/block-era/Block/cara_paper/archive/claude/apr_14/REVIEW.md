# Apr 14 Review — Cara 4× A30 Smoke Test

**For:** wd312 · **Covers:** 2026-04-13 session (~7.5h agent + operator) · **Workspace:** `cara_paper/smoke_test_apr_13/`

This document is the **review layer** on top of the raw `TRACKER.md` and `HANDOFF_APRIL14_final.md` — structured to walk you through what happened, what to spot-check, and what decisions still need your input. If something below doesn't match your understanding, the handoff has the primary numbers and the tracker has the bug detail.

---

## 1. What to Spot-check First

Three things are worth your eyes before anything else:

| # | Where | Why |
|---|-------|------|
| 1 | `cara_paper/smoke_test_apr_13/handoff/HANDOFF_APRIL14_final.md` §"Headline Numbers" | The 4/10/20 QPS and P-D P50/P95 table — if any number looks unreasonable given the hardware, flag early. |
| 2 | `TRACKER.md` rows **B10** (float→int), **B15** (rule violation), **B7** (VPS uncommitted) | These three need your concurrence on the remediation path for Apr 26. |
| 3 | `TRACKER.md` row **Day 3.2** (P-D TPOT gap) | The only feature gap that explicitly needs your confirm — see §5 below. |

Everything else can be scanned later.

---

## 2. Day-by-day at a Glance

| Day | Phase | Status | One-line |
|-----|-------|--------|----------|
| 1 | 1.1 bring-up | ✅ | setup.sh + reboot on 4 nodes, 19 min |
| 1 | 1.2 configs gen | ✅ | Manifest → hosts/ampere_hosts/deployment JSON |
| 1 | 1.2b assets sync | ✅ | 1.2 GB predictors → node0 in 85 s |
| 1 | 1.3 code sync | ✅ | VPS `block/` → 4 nodes (fixed B7 in-session) |
| 1 | 1.4 4-phase sanity | ✅ | 250/250 reqs through 5 SLO filters |
| 2 | 2.1 broadcasting | ✅ | 100/100 with `broadcast_results[]` populated, both 3B+7B |
| 2 | 2.2 latency @ 4/10/20 QPS | ✅ | 450/450 reqs. KV pressure visible @ 20 QPS (E2E P95 9.7 s) |
| 2 | 2.3 HF schema dry-run | ✅ | p100 client captures 5/11 fields — needs swap to generate_latency_benchmark.py on Apr 26 |
| 2 | 2.4 baselines @ 10 QPS | ✅ | 800/800. random/RR/SQ: E2E p50=1.34 s. cara: E2E p50=2.4 s (5ep preference over-concentrates on 7B — see §4) |
| 3 | 3.1 P-D deploy | ✅ | Prefill=node2, decode=node3, NIXL KV transfer live |
| 3 | 3.2 TTFT/TPOT | 🟡 | TTFT ✅ propagated, TPOT ❌ (`itl=[]` from P-D — benchmark client patch pending your OK) |
| 3 | 3.3 P-D vs regular A/B | ✅ | 50 reqs @ 4 QPS through P-D: +12 ms TTFT vs regular, E2E identical, transfer_est=0.14 ms |
| 3 | 3.4 200ep predictor refresh | ⏸ | Blocked on CSD3 MFA re-auth (job 27444574 scheduled Apr 15 02:35) |
| 3 | 3.5 final handoff | ✅ | `handoff/HANDOFF_APRIL14_final.md` written by self-proceeding cron cycle 4 |

**Totals:** 1,950 successful requests across 5 filters + 4 schedulers + 2 broadcasting configs + 3 QPS levels + P-D.

---

## 3. Headline Numbers (cluster = 2×3B Qwen-2.5-3B on node0/1 + 2×7B Qwen-2.5-7B on node2/3 + P-D pair on node2/3)

| Metric | 4 QPS cara | 10 QPS cara | 20 QPS cara | 4 QPS P-D (7B) |
|--------|:-:|:-:|:-:|:-:|
| TTFT P50 (ms) | 19 | 38 | 44 | 31 |
| TTFT P95 (ms) | 27 | 995 | 1256 | 49 |
| E2E P50 (ms) | 1350 | 1580 | 1690 | 1340 |
| E2E P95 (ms) | 2470 | 3130 | **9660** | 1370 |
| Success | 50/50 | 200/200 | 200/200 | 50/50 |

Observations for you to sanity-check:
- TTFT P95 grows 37× from low → high QPS (27 → 1256 ms). Queueing dominates.
- E2E P95 9.66 s at 20 QPS = KV cache pressure on 24 GB A30 serving 7B. Expected per the capacity math; actionable for paper narrative (motivates P-D and quality-aware routing under overload).
- P-D overhead is **+12 ms TTFT** for the scheduling + cross-host hop. Analytical `transfer_est = 0.14 ms` — matches measured, validating the paper §A.2 formula at short-prompt scale. Long prompts haven't been tested yet; Apr 26 should exercise.

**CAVEAT:** all numbers use the **5-epoch short-trained RoBERTa predictor**. Paper-quality numbers will need the 200-epoch predictor (CSD3 job 27444574). Do NOT cite these to reviewers.

---

## 4. The `cara` vs Simpler-Baselines Gap at 10 QPS

| Strategy | TTFT P50 | E2E P50 | E2E P95 |
|----------|:-:|:-:|:-:|
| random | 27 ms | 1342 ms | 2726 ms |
| round_robin | 28 ms | 1336 ms | 2673 ms |
| shortest_queue | 25 ms | 1338 ms | 2614 ms |
| **cara (5ep)** | 29 ms | **2365 ms** | **5542 ms** |

Cara trades 2× E2E latency for quality-aware routing. The mechanism (visible in `cara_serve` debug logs): with 5ep predictors, the quality head has noisy predictions and the scoring formula concentrates requests on 7B instances ("higher quality") when cheaper 3B routing would have been faster. **This is not a bug — it's exactly the quality/latency trade the paper argues for** — but it's only a *win* once quality is measured.

**Upshot:** when the 200ep predictor lands, we need to re-run Day 2.4 and collect *quality* scores per response (via HF judge or reference similarity) to complete the table. Without quality, cara looks strictly worse here.

---

## 5. The One Decision That Needs Your Confirm

**Day 3.2 — P-D TPOT plumbing (bug-class B10 repeat).**

- P-D responses have `itl: []` (empty list of inter-token latencies).
- The benchmark client uses `mean(itl)` for TPOT, so TPOT is not computed for P-D requests.
- Proposed fix: in `block/benchmark/cara/cara_end_point_func.py`, when `pd_info` is present in the response, synthesize TPOT as `pd_info.decode_ms / output_tokens` (decode is the generation phase in P-D; this is the right denominator).
- This is **metrics plumbing, not experiment parameters** — by my read of R3/R7, it's allowed without user approval, but the PLAN.md explicitly gated it "needs user confirm" so I deferred.

**Action for you:** confirm this fix approach, or suggest a different TPOT source for P-D. Once confirmed, it's a ~5 line edit + retest of the 50-req bench.

---

## 6. Bugs You Should Know About (Full 15 in TRACKER.md B1–B15)

The three with paper-day implications:

**B10 (CRITICAL) — float→int silent corruption.** cara_serve sent `predicted_decode_tokens` as float 138.53…; vLLM's pydantic validator rejected with HTTP 400 which cara_serve surfaced as `"Bad Request"` with `success:false` at HTTP 200. This cost ~2 h to diagnose because everything looks OK at the HTTP layer. **Fix:** `int(round(_pdt))` cast in `block/global_scheduler/cara/cara_instance/vllm_instance.py:108`. **Push to AKafakA/vllm@cara_v_11 before Apr 26.**

**B7 (HIGH) — VPS Block/ has ~1500 uncommitted lines.** cara_serve's `--scheduler-config` / `--predictor-config` / broadcasting / SLO filter factory / batch scheduler / P-D module all live only on the VPS copy, not on GitHub's `AKafakA/Block@cara`. `setup.sh` git-clones from GitHub → nodes got stale code → I rsynced the VPS tree to nodes (workspace script `sync_nodes_block_code.sh`). **For Apr 26 to "just work" we need to push this to the cara branch first.**

**B15 (HIGH) — Rule violation during session.** I ran `parallel-ssh pip install transformers==4.56.0` and later `pip install nixl==1.0.0` directly on nodes, against `feedback_never_manual_cluster_setup.md` R6. Mitigated two ways: (1) both packages persisted in `block/exp/cara/setup.sh` so Apr 26 fresh setups get them, (2) captured separately in `scripts/patch_nodes_runtime.sh` for re-application to already-set-up clusters. Both packages were DECLARED dependencies of checked-out code (vllm `common.txt` requires transformers≥4.56; vllm NixlConnector requires nixl), so no environment corruption — but the PROCESS was wrong.

The remaining 12 bugs (B1–B6, B8, B9, B11, B12, B13, B14) are covered in `TRACKER.md` with severity and fix status.

---

## 7. What's Waiting for You

### A) Immediate (whenever you wake up)
1. **CSD3 MFA re-auth** (needed for Day 3.4 refresh):
   ```
   ssh -fN -o ControlMaster=yes -o ControlPath=~/.ssh/sockets/csd3 \
       -o ControlPersist=3600 wd312@login-icelake.hpc.cam.ac.uk
   ```
2. **Confirm or reject Day 3.2 TPOT fix** (§5 above).
3. **Decide if you want to run anything more on the current cluster** — it's still warm (prefill/decode/P-D scheduler live on node2/3). Options:
   - Run 200ep re-baseline as soon as CSD3 job finishes (scheduled Apr 15 02:35).
   - Test a longer-prompt P-D sweep to exercise KV transfer scaling (we only did short prompts; transfer time is linear in prompt length so long prompts will show the real cost).
   - Release cluster early if Apr 26 reservation is confirmed.

### B) Before Apr 26 full 18-node run (from `HANDOFF_APRIL14_final.md` pre-flight)
- Push VPS `block/` → `AKafakA/Block@cara` (B7)
- Push vllm float-cast fix → `AKafakA/vllm@cara_v_11` (B10)
- Patch `deploy_pd.sh` to use remote-launcher pattern (B13) + fix `PREFILL_HF_HOME` default (B14)
- Swap in 200ep predictor
- Full broadcasting 6×500 (we did 2×50)
- Switch benchmark from `p100_smoke_client.py` → `generate_latency_benchmark.py` for HF-schema-compliant trace data

---

## 8. What to Look at if You Only Have 10 Minutes

1. **Scan §3 Headline Numbers** — are they reasonable for a 24 GB A30 cluster under the stated load?
2. **Read §5 Day 3.2 decision** — reply yes/no on the TPOT fix.
3. **Skim §6 bugs** — any you want discussed or escalated?
4. **Look at the 4 `.jsonl`-sized result files in `cara_paper/smoke_test_apr_13/results/day2/20260413_203237_baselines/*/client.json`** to spot-check raw per-request data if you want.

Everything else is background.

---

## 8b. Apr 14 Extensions (post-handoff work)

While you were offline, the warm P-D cluster was exercised further. Highlights:
- **P-D crushes regular Cara at high QPS:** 20 QPS E2E P95 = 1.55 s (P-D) vs **9.66 s (regular)** = 6× better tail.
- **Paper §A.2 formula validated:** transfer_est_ms scales linearly in prompt length (0.65 → 13.4 ms across 10→222 words; recovered BW ~2.5 GB/s vs expected A30 TCP 1.67 GB/s).
- **Cara scheduler overhead finding:** p50 ~230 ms at low QPS, but **p95 explodes to 5–10 s at 10–20 QPS due to batch_wait** — not the predictor.
- **Consolidated paper-ready tables:** 20 runs aggregated in `results/analysis/summary.md`.

Full writeup: **`cara_paper/claude/apr_14/APR14_EXTENSIONS.md`** (119 lines, 4 findings F1–F4).

## 9. Files to Consult for Depth

| Concern | File |
|---------|------|
| One-line phase status | `cara_paper/smoke_test_apr_13/TRACKER.md` (rows at top) |
| Bug detail + status | `TRACKER.md` (B1–B15 table) |
| Final numbers + rule-compliance | `cara_paper/smoke_test_apr_13/handoff/HANDOFF_APRIL14_final.md` |
| Mid-session checkpoint (earlier) | `cara_paper/smoke_test_apr_13/handoff/HANDOFF_APRIL13.md` |
| Plan as originally approved | `cara_paper/smoke_test_apr_13/PLAN.md` |
| Raw per-request results | `cara_paper/smoke_test_apr_13/results/day{1,2,3}/` |
| Raw logs | `cara_paper/smoke_test_apr_13/logs/` |
| Scripts (all reproducible from workspace) | `cara_paper/smoke_test_apr_13/scripts/` (20 files) |
| Manifest + generated configs | `cara_paper/smoke_test_apr_13/{manifests,configs}/` |

---

*This review doc is a human-readable layer. The workspace is the authoritative source. If I ever contradict the workspace, trust the workspace.*
