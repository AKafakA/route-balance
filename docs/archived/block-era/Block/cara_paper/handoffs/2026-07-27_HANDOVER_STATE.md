# HANDOVER STATE — 2026-07-27 21:45Z (model switch Fable→Opus; autopilot loop continues)

## TL;DR
The rebuttal packet is **FROZEN + AUTHOR-APPROVED** and waiting for the author to paste it into
OpenReview. Gemma generation is **live on node0 (~46%, ETA ~22:15Z)**. Nothing is blocked on
analysis; the two open work streams are (a) the author's posting action, (b) the Gemma morning
pipeline. The autopilot loop is armed (next wakeup ~22:38Z).

## 1. THE DELIVERABLE (ready to post — do not re-edit without author instruction)
`cara_paper/handoffs/2026-07-27_replies_FINAL_FORMAT.md` — 5 posts, Hymba-style
(POST 0 General Response + meta-review · POST 1 = 287j · POST 2 = gTfM · POST 3 = faAU ·
POST 4 = AC comment). Header carries the ✅ AUTHOR-APPROVED stamp (21:20Z). Final gate sweep
0-hit on: SIGNOFF/STOP-ITEM remnants, codename, links, k=50, stale numbers (18216/21558/11392),
square-wave/40-6, basis language, version language, placeholders.
Supporting docs: `2026-07-27_ACTUAL_REVIEWS_VERBATIM.md` (authoritative reviewer mapping:
**287j = clarity/batching (rating 3) · gTfM = estimator Q1–Q4 (4) · faAU = generality/cold-start/
dispatchers/tuning (4)**) and `2026-07-27_REBUTTAL_FORMAT_GUIDE.md` (style rules + score strategy).
Internal frame (NOT posted): `2026-07-27_rebuttal_draft_noGemma.md`.

## 2. LIVE PROCESSES
- **node0 = `asdwb@d8545-10s10505.wisc.cloudlab.us`** — Gemma broadcast generation
  (18,549 prompts × 4 tiers), started 21:04:10Z, ~46% at 21:43Z, ETA ~22:15Z.
  Watch: `/tmp/_gemma_gen.log` + `experiment_output/jul25_stage_a/gemma_family/gen.bench.log`
  (progress bar uses \r — pipe through `tr "\r" "\n"`). Marker on completion: `/tmp/_gemma_gen.done`.
- Gemma ladder healthy **14/14**: 1B×3 + 4B×5 on A30s (`node_hosts` path), 12B×4 single-A100 +
  27B TP2×2 on the d8545 pair (`instances` path, ports 8003/8004 and 8001).
- Cold-start clock **T0 = 19:49:50Z** (`$S/_gemma_T0`) — this is faAU's measured onboarding
  wall-clock; it deliberately includes the dtype failure + fixes (honest).
- VPS scratchpad `S=/tmp/claude-1000/-home-wd312-Code-llm-Block/cae98b37-367c-48b6-8819-5f1f4689973e/scratchpad`.

## 3. NEXT STEPS (task #7)
On `/tmp/_gemma_gen.done`: pull outputs + two-location sync → build per-model scored corpus →
llama-8b judging (see `window_setup_work/gemma_pipeline/GEMMA_PIPELINE_PLAN.md`) → **G2.5
ladder-separation gate** (if tiers don't separate: STOP + author choice) → static priors + KNN +
training → G5 sampled cells → family table as an **in-window follow-up comment** (faAU's W1 =
the conditional score raise). Fallback chain if the family stalls: Gemma-2 → Llama → 5k subset.

## 4. TONIGHT'S GEMMA BUGS (all fixed — do not re-hit)
1. **fp16 hard-rejected** for gemma3 by vLLM ("numerical instability") → whole ladder is **bf16**
   (author-approved redesign; V100/c4130 can't do bf16, hence 12B moved to A100 singles).
2. **Stale Qwen `VLLM::Worker` engine cores** held all 8 A100s (processes retitled → invisible to
   name-pattern pkill) → kill by PID from `nvidia-smi --query-compute-apps=pid`.
3. `deploy_cara.py` needs **`--hf-token`** (gated repos 401 on live chat_template fetch).
4. Broadcast list needs **HF names** (`google/gemma-3-1b-pt`), not config keys.
5. **`--chat` routes to the backend chat endpoint**, which gemma-pt rejects (no chat template).
   Author-ruled fix implemented + committed (`970fdd5`): family-gated branch in `cara_serve.py` —
   `--chat --model-family gemma` applies the Gemma turn template scheduler-side and sets the
   `<end_of_turn>` stop; **Qwen default path byte-unchanged**. Bench also needs
   `--skip-chat-template` (our shim's flag name; the upstream spelling parses but isn't wired).
   Stop behaviour verified (192/512 tokens, natural halt); era sampling confirmed injected
   server-side (temp 0.0, freq penalty 1.2).

## 5. BANKED GUIDANCE FOR THE V3 (topology) FOLLOW-UP POST — read before drafting it
- **Do NOT post the ss532 λ30 flip row as a plain −2.3%**: that cell is scheduler-overhead
  dominated and non-stationary (overhead still climbing at run end); serving-side gain is −36.9%,
  consistent with the −28..−31% band at λ6–24. Either replicate it or scope it explicitly.
- **Ordering claims must be system-level only.** The argmax *dispatcher* ordering INVERTS on topoB
  (RR < Po2 < SQ at λ24/30; SQ worst by 22% at λ30) vs SQ-best on the main topology.
- Correct original-topology passthrough-Po2 counterparts live in `po2/passthru_po2_l*`
  (NOT `po2_routers/`, which holds br4 rows).
- Flip argmax column: 12/12 banked, 0 fails, code identity vs the Jul-26 flip cells PROVEN
  (file-hash match node0↔VPS; last code change Jul-25 19:40, pre-flip; topoB config md5 identical
  across both deploys).

## 6. STANDING RAILS (binding, author-set)
No links · no codename ("cara") in posted text · no k=50 statements · **no version/build ("v2")
language — this is a v1 rebuttal, hide v2 totally** · one RB preset per table (uniform default) ·
v1exact basis for every paper-comparable number · current-tree tables relatives-only · never
re-assert BEST-Route-frontier or t05 framings · Pareto-shaped claims only vs single-model
references · quality margins ≤0.01 only with computed CIs (`window_setup_work/ci_results_jul27.json`
+ `ci_ext_results.json`) · burst support = standard gamma mode only (sq40/6 + burst-widen cells
r28/r29/g05/g075 stay internal, truthful-if-asked) · **presentation gate: nothing posts without
author sign-off** (given for the current packet) · never change experiment params to work around
errors — STOP + report.

## 7. KEY NUMBERS (verified, in the posted packet)
- γ0.25 r30, 6 paired runs: learned 3156 vs static 3391, **mean −6.7%** (per-run −15.28, −5.03,
  **+6.69**, −11.77, −9.93, −4.79; 5/6 favorable), 95% CI [−14.8%, +1.4%]. Author re-confirmed
  the "consistently directionally favorable" wording with these signs in hand (21:20Z).
- Stationary λ6–24: parity ±0.8%; λ30 static ahead +6.7% (reported honestly).
- Arm2 (no latency term): mix frozen 69.1% smallest tier at every rate; quality drops
  cell-level 0.014–0.018 (paired 0.008–0.012, CI at λ12 [0.002, 0.019]) — below BR4.
- V2: BR4+Po2 knee 2636→18273 (λ12→18), λ24 48182, λ30 54112; RB 20.0× lower at λ24, 16.5× at λ30.
- V4: prompt-blind −0.013 global; ladder within 0.005 from 25% coverage; holdout globals ≤0.006
  (gsm8k a tie); excluded-domain slices statistically unchanged (CIs span zero); e2e flat ±2%.
- §3.3: always-72B 0.533 / 14B 0.508 / 7B 0.501 / 3B 0.473; presets 0.505–0.512, **all
  statistically indistinguishable from always-14B** (paired CIs span zero) — pure tie language.

## 8. AUDIT STATUS
Two multi-agent audits complete (44-agent packet review + 23-agent final Qwen check): **49 + 31
findings all dispositioned; zero cells invalidated; zero reruns required** — every defect was
documentation-level. All confirmed items applied; CI extensions computed on dev-gpu closed the
last uncovered margin claims.
