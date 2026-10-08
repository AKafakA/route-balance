# OVERNIGHT HANDOFF — 2026-07-28 ~02:45Z (author to sleep; autopilot loop armed)

## ⛔ THE ONE THING THAT NEEDS YOU (everything else is automatic)
POST 3's onboarding-cost sentence cites a judge throughput that was measured with the **wrong
judge**. `≈24 calls/s ⇒ ≈13 min/model column` came from the **gemma-3-12b-it judge-ALT**
(2026-06-07_v2_rerun_handoff.md:125); EXECUTION_HANDBOOK.md:148 mis-labelled it "Llama-3.1-8B"
and the packet inherited it. Metric-of-record measured rate (Llama-3.1-8B, 4xA100 sharded bf16)
= **~4.2 pairs/s** ⇒ **~74 min/model column**. 5.7x worse, so it is your call, but the current
sentence cannot survive "how did you measure that?". Ready replacement text + a DO-NOT-POST
banner are at the top of `2026-07-27_replies_FINAL_FORMAT.md`. Nothing else in the packet is
affected — that figure appears once.

## RUNNING NOW
- **Gemma generation**, node0, relaunched 01:50:28Z through the REPAIRED `gemma_night.sh generate`
  (not a fork). Path is now **identical to Qwen**: `--chat --broadcasting`, no `--model-family`
  (verified by execution that gemma-3-12b-it accepts the backend chat endpoint: "391",
  finish_reason "stop"). 2,720/18,549 @ 2.31 it/s at 02:05Z -> **ETA ~04:00Z**.
- **Judge lease setup** (3x d7525, separate lease): apt full-upgrade SUCCESS all 3 at 02:04Z;
  CUDA 12.8 + nvidia-open installing; driver needs a reboot before nvidia-smi works.

## AUTOMATIC OVERNIGHT SEQUENCE (no decisions needed)
Main cluster and judge lease are SEPARATE hardware, so these run in parallel:
1. gen done ~04:00Z -> pull + two-location sync -> **rebuild the truncation filter from the -it
   outputs** (the banked 41.7% filter is -pt-derived and must not be reused)
2. [main] teardown Gemma -> topoB Qwen deploy (~1.7h) -> **8 v1exact topology cells λ12/λ24**
   -> **TOPOLOGY BANKED ~07:30Z = the guaranteed fallback deliverable**
3. [judge lease] both chains from ~04:15Z: deepeval TEST->TRAIN, rubric TRAIN->TEST
   (per your both-keys-both-splits ruling). A100s join once topology finishes.
4. gating legs ~09:30Z -> priors/KNN/BR4 training ~10:15Z -> Gemma redeploy ~11:45Z
   -> **G5 cells λ12/λ24 ~13:15-14:45Z**. **HARD DEADLINE 14:00 BST = 13:00 UTC** (author, 2026-07-28 ~03:00Z).

**Derisking holds automatically:** topology banks at ~07:30Z regardless of Gemma. If Gemma dies at
the separation gate, at training or at redeploy, the topology block posts and nothing is lost.

## I WILL STOP AND WAIT (not improvise) IF
RULE-0 / cell_audit mismatch · completed!=3534 or failed>0 · OOM or crash · any anomalous number ·
anything that would require changing an experiment parameter. Auto-fix stays limited to
infra-class only (dirs, PYTHONPATH, stale process kill, transient network retry, permissions).

## MEASUREMENTS BEING CAPTURED for faAU's training-time answer
Generation wall-clock · judging (REPORTED at the clean 4xA100 rate; the A30 lease is opportunistic
capacity only and never enters the reported arithmetic) · KNN + priors + BR4 head training.
Detail + corrected provenance table: `window_setup_work/gemma_pipeline/onboarding_timings.md`.

## OPEN, NOT URGENT
Packet-audit disposition (9 blocking / 21 major / 25 minor from the earlier run) · λ30 disclosure
wording (2526 vs 3284) · Phase B purge · public-repo correction commit (LOCAL only) ·
`block/config/hosts` is STALE (lists d7525-10s10335, reallocated to the judge lease) — cleanup
after the window; setup.sh is already isolated behind a HOSTS_FILE override.

## UPDATE ~02:55Z — the STOP-ITEM is CLOSED; packet is postable
Author resolved it: onboarding cost is a report, not a thesis. POST 3 now carries the
metric-of-record rate (≈4.2 pairs/s, Llama-3.1-8B on 4×A100 ⇒ ≈74 min/model column) and the
author's reframing — thank the reviewer, state the four-tier Qwen corpus IS publicly released
(verified on disk: 14,919/3,634, each model carrying `response` + both judge keys +
`reference_text`), and commit to adding the second-family corpus.
**Nothing in the packet now depends on the Gemma track completing.** No repo name, no link
(NO-LINKS rail + review anonymity). There are no open author decisions.

## UPDATE ~03:05Z — HARD DEADLINE SET: 14:00 BST = **13:00 UTC**
Author set the hard submission deadline at 14:00 BST (13:00 UTC), ~2 h after the Phase-1 soft
deadline. Plan checked against it:
- **Topology banks ~07:45Z — 5.25 h of margin. The packet is postable from that point.** ✅
- **G5 / Gemma family lands ~13:45-15:15Z — MISSES the hard deadline by ~1-2 h.** ❌
  This breaks nothing: POST 3 already scopes it ("results are in progress in this discussion
  window and will follow as a separate comment"), and the author-posting window runs to Aug 3.
  So the 14:00 BST submission carries Qwen + topology; Gemma follows in-window as written.
- Only way to pull Gemma inside 13:00Z would be to DROP topology so judging gets the A100s from
  ~04:55Z (G5 ~10:45-12:15Z). NOT recommended and not taken: a fourth Gemma failure would then
  leave us with neither deliverable. Topology-first keeps a guaranteed generality answer.
- Generation rate drifting DOWN (2.31 -> 2.08 it/s; 6,196/18,549 at 03:0xZ) -> gen done ~04:40Z,
  ~40 min later than the previous estimate. Downstream ETAs above already include this.

## UPDATE ~03:15Z — MAXIMUM DEADLINE 16:00 BST = **15:00 UTC**; SPLIT-DEPLOY OPTIMIZATION ADOPTED
Author: 14:00 BST (13:00Z) = target, **16:00 BST (15:00Z) = maximum acceptable**. G5 at
13:45-15:15Z was too close to the ceiling, so the Gemma redeploy is being SPLIT rather than run
as one 1.5 h block on the critical path:
- the Gemma **A30 tiers** (1B x3, 4B x5) deploy as soon as topology frees the A30s (~07:45Z),
  IN PARALLEL with judging — judging only occupies the A100s, so there is no contention;
- only the **A100 tiers** (12B x4 + 27B TP2x2, ~40 min) deploy after judging ends (~10:00Z).
=> ladder ready ~10:40Z instead of 12:15Z; training completes ~10:45Z; **G5 runs 12:15-13:45Z**
— inside the 15:00Z ceiling in every case, and inside the 13:00Z target in the good case.
Topology is UNAFFECTED and still banks ~07:45Z as the guaranteed deliverable.
NOTE this is a scheduling change only — no experiment parameter, config, or model changes.
Deploy still goes through deploy_ladder.sh (which already handles node_hosts and instances tiers
separately, so a tier subset is a natural invocation, not a new script).

## UPDATE ~03:30Z — WORST-CASE RULE ENCODED (author) + judge GPUs are LIVE
**Author's fallback:** if the A30 judge lease is not working or is too slow, do NOT redeploy Qwen
after generation — give the **A100s to Gemma judging immediately** and drive Gemma to completion
as fast as possible, because "even Gemma standalone could fill all gaps" (it answers generality on
BOTH the family and deployment axes AND supplies the measured onboarding/training-time example).

Encoded as a hard gate that fires ONCE when generation completes, before any topoB deploy:
`window_setup_work/gemma_pipeline/a30_viability_gate.sh`
- **PASS -> PATH A** (current plan): topoB + 8 v1exact topology cells banked ~07:45Z; judging
  starts on the A30 lease; A100s join when topology finishes; split Gemma redeploy; G5 12:15-13:45Z.
- **FAIL -> PATH B**: skip topoB entirely, A100s to judging at once, Gemma ASAP, topology deferred.
- FAIL if any node lacks a working nvidia-smi, the judge model will not load/serve, or aggregate
  smoke throughput < 0.5 pairs/s (below that the lease is noise, not help).

### ⚠ PATH B HAS A PACKET CONSEQUENCE THAT MUST NOT BE MISSED
POST 3's W1 still contains the topology table built from the **14 CURRENT-TREE cells**, which
contradict W3's v1exact BEST-Route numbers (48,182 ms vs 2,296 ms at λ24, same label, same rate).
On PATH B that block is never rerun, so it **MUST BE DELETED** and the Gemma family result becomes
the W1 generality answer. Posting it unchanged is the one outcome worse than posting neither.

### STATE AT 03:30Z
All **3 judge nodes have working A30 drivers** (24,576 MiB each — confirms A30 = 24 GB, not the
48 GB in CLAUDE.md; Apr-13 record was right). No reboot was needed. setup.sh continuing with the
vLLM/Block install. Gate dry-run: GPUs present on all three -> Path A on track pending the
throughput smoke.
Gate script bug found and fixed on the dry-run: `ssh` without `-n` swallowed the hosts file from
stdin, so the read-loop exited after ONE host and the gate "passed" having checked a single node.
