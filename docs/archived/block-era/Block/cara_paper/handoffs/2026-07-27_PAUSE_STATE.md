# PAUSE STATE — 2026-07-27 (author order: "pause here and terminate the current chain")

## What is stopped
- EXP2 runner + in-flight benchmark: killed at cell boundary state (5/16 banked, all complete
  jsons — the newest is whole; nothing partial on disk). All VPS watchers: none running.
- Chain monitor: stopped. No automation will launch anything until an explicit author go.
- Cluster serve: one idle cara_serve may remain on node0 (harmless; next runner restarts it).
- Off-cluster (harmless, left running): dev-gpu judge-weight prefetches + uv env (READY).

## Banked & verified as of pause (~14:0xZ)
- v1exact: A0/retests 4 · weight grid 17 · bridges 26 · PT 17 · BR4 8 · AP core 8 = 80 cells
  (+2 br4_sq variance runs). All audited; twins bit-exact; calm-regime checks logged.
- Current tree: four-arm 28 · burst 37 · overload 2 · P4 32/32 · EXP2 5/16 · flip 25 banked ·
  po2 17 · router corroboration 27. All synced two-location.
- Draft: 2026-07-27_reply_drafts_ASSEMBLED.md (verified, 4 review items flagged) + frame with
  V1/V2 filled, V3/V4 rails applied (relatives-only, ladder-only). Handbook rulings #1-7 locked.

## To resume (each = one command, scripts staged on node0/VPS)
1. EXP2 remaining 11: `setsid nohup bash ~/exp2_cells.sh` on node0 (skip-existing resumes).
2. Flip argmax col ×12: topoB redeploy (deploy_cara jul27_topoB + health) → `flip_argmax_cells.sh`.
3. Gemma: checkpoint go → gemma_night.sh path (configs+corpus staged; smoke kit ready:
   gemma_smoke_200.jsonl + mini deployment template awaiting side-node hostnames).
4. Overnight: `qwen_overnight.sh` on node0 post-Gemma (pw-sweep ×28; needs Qwen redeploy first).
5. Held-live re-base (task #6): P4 ×32 + EXP2 ×16 on v1exact (~5h).
Timing note: rebuttal button open to Jul-27 AoE (≈Jul-28 11:59Z); Official Comments equivalent
to Aug-3 (PC-confirmed). Lease ends Jul-31 19:00.

## Open author decisions at pause
- Gemma go (checkpoint) · side-node hostnames (smoke) · draft review (4 flagged items) ·
  held-live core-vs-optional re-base scope.
