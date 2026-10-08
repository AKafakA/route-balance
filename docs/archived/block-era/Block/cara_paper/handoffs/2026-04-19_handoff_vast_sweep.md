# 2026-04-19 Handoff — Vast HP Sweep (honest record)

**Status at write time:** sweep running, 3 prior failed attempts, findings doc at `cara_paper/apr19/sweep_v4_findings.md` being updated every 20 min by cron `63991c7a`.

## What was done today

**Hardware**: Vast.ai 2× V100-PCIE-16GB rented at 11:22 UTC. ControlMaster SSH alias `vast` set up in `~/.ssh/config` (12 h persist, TCPKeepAlive). Persistent remote tmux session `cara` on window 0 (bash) and window 1 split (GPU 0 / GPU 1).

**Codebase change landed** at `block/predictor/cara/offline_training/train_fused_bert_predictor.py`:
- Added `--weight-decay` CLI arg (default 0.0, matches prior behavior)
- Added `--warmup-ratio` CLI arg (default 0.03, matches prior behavior)
- Threaded both through to `TrainingArguments(...)`

Previously `warmup_ratio=0.03` was hardcoded and `weight_decay` used HF default (0.0). Neither was CLI-exposed before today. This unlocks HP sweep over these knobs.

## Runs executed (all on Vast today)

| Run | Config | Wall | Outcome | Artifacts |
|---|---|---|---|---|
| 1 | patience=3, lr=1e-5, poly, wd=0 (parallel) | ~7 min | Crashed (disk full at epoch-3 save) | None |
| 2 | patience=3, `save_total_limit=1` (parallel) | ~7 min | Crashed (same — concurrent save burst still blew disk) | None |
| 3 | patience=3, save_total_limit=1, after clearing `/root/.cache` | ~20 min | Succeeded | length_bucket acc=0.4232, deepeval mae=0.2257 — then **deleted at user request** because patience=3 was the wrong direction |
| 4 | no early-stop, 200 epochs (parallel) | killed after ~10 min | Killed by user (wrong direction vs HP sweep goal) | None |
| 5 | sweep V1 (LR × scheduler only) | killed ~90 s | Killed to expand sweep | None |
| 6 | sweep V2 (+ wd, warmup) | killed ~90 s | Killed to add running-best retention logic | None |
| 7 | sweep V3 (running-best retention) | DIED SILENTLY at config 1 transition — quoting bug in embedded Python helper | **Baseline artifacts stuck on vast, later deleted** |
| 8 | sweep V4 (current) | running | launched 19:16 UTC, in progress | updated continuously in apr19/sweep_v4_findings.md |

## Budget burned (honest)

~6 h of Vast rent at ~$0.25/h = **~$1.50 total**. Breakdown:
- ~$0.15 — Run 3 (patience=3 success, then deleted because wrong direction)
- ~$0.10 — two disk-full crashes
- ~$0.06 — killed 200-epoch run
- ~$0.03 — sweep V1/V2 brief launches
- ~$0.85 — **silent dead V3, 3 h idle before detected** (the largest single waste)
- ~$0.25 — current V4 sweep (ongoing, ~1.5 h in at handoff time)

Apr 18 CSD3 artifacts (`roberta_fused_200ep/`) remain the reference of record until V4 completes. Nothing on openclaw today beats them yet.

## Why V1–V3 failed (so we don't repeat)

1. **Concurrent-save disk math**: Each HF checkpoint is ~1.4 GB (model + AdamW + scheduler + scaler), not 500 MB as initially estimated. Two parallel trainings saving at the same epoch boundary transiently hold 6 ckpts on disk before rotation. 16 GB disk minus ~7 GB baseline = 9 GB, minus transient 6 × 1.4 = 8.4 GB → zero margin.
2. **Clearing `/root/.cache`** reclaimed ~3 GB pip cache, buying enough margin for concurrent training with `save_total_limit=1`.
3. **V3 quoting bug**: embedded Python inside a bash function inside an ssh heredoc with single quotes — inner `'accuracy'` got interpreted as bash quote boundaries, stripping the quotes in the file. Python saw `key = accuracy if  == length_bucket else mae` → SyntaxError. `set -e` exited cleanly. Outer tmux wrapper's `| tee && echo LB_SWEEP_DONE` pattern printed the false success marker regardless of script exit code. The old monitor's failure regex didn't include `SyntaxError`, and it used `LB_SWEEP_DONE` as the done flag — so it would have reported SWEEP_ALL_COMPLETE on a silent death.

## What sweep V4 adds

- **Standalone `extract_metric.py`** on vast (no embedded Python in bash).
- **Sentinel files** at `/workspace/Block/sweep_state/<target>.status`: orchestrator's EXIT trap writes `SUCCESS` or `FAIL exit=N stage=S config=C` on exit. Monitor/cron can distinguish real success from false-success.
- **`rsync --mkpath`** — creates missing parent dirs on openclaw. V1–V3 would have failed here too even without the Python bug.
- **`set -o pipefail`** so pipe-to-tee doesn't hide training failures.
- **Running-best retention**: only the winning config per target keeps its 476 MB `fused_model.pt` on openclaw. Others keep only 16 KB of metric JSONs. Worst-case openclaw footprint ~970 MB for the whole sweep.
- **End-to-end dry run executed first**: mini-orchestrator with 1 epoch + 100 rows per config ran before the real launch. Caught the `--mkpath` bug (rsync IO error 11). Re-ran clean, all markers fired, artifacts correctly on openclaw, cleanup correct.

## What the cron loop does now

Fires every 20 min (`63991c7a`, `7,27,47 * * * *`). Each iteration:
1. Polls vast sentinels + best-state files + disk/GPU
2. Pulls any newly-landed `training_results.json` + `log_history.json` from openclaw
3. Updates `cara_paper/apr19/sweep_v4_findings.md` with completed-config comparison table + trend observations
4. Detects common infrastructure bugs (disk running low, stuck process — training log not touched >15 min, missing artifacts after FINISHED marker) and applies only **non-semantic** quick fixes (e.g. retry transient ssh, clear caches, make missing directories). **Never touches experiment parameters.**
5. On sentinel FAIL → report exit code + stage + config name, do NOT retry
6. On both sentinels SUCCESS → summarize winners, propose next sweep, stop loop

## Key paths to remember

- Sweep orchestrator (vast): `/workspace/Block/sweep_target.sh` (99 lines), `/workspace/Block/extract_metric.py`
- Training script (VPS): `block/predictor/cara/offline_training/train_fused_bert_predictor.py` (now has `--weight-decay`, `--warmup-ratio`)
- Vast sentinels: `/workspace/Block/sweep_state/{length_bucket,deepeval}.status`
- Vast per-target best-metric state: `/workspace/Block/sweep_state/{length_bucket,deepeval}_best.txt`
- Landed artifacts (openclaw): `~/Code/llm/Block/models/cara/roberta_fused_vast_sweep/<target>/<config>/`
- Vast logs: `/workspace/Block/logs/sweep_<target>.log` (orchestrator), `/workspace/Block/logs/<target>_<config>.log` (per-config training)

## Integrity rules enforced

- No per-model Spearman ρ (per `feedback_no_spearman_rho.md`). Use avg MAE + BestAcc + RankCorr.
- `save_total_limit=1 + load_best_model_at_end=True + metric_for_best_model=accuracy|mae` keeps the true best. Verified against today's patience=3 run.
- R3/R7: experiment parameters (lr, batch, max_length, wd, warmup, scheduler) are only changed via explicit user approval. The sweep IS the user-approved variation.
- No silent retry on failure. Errors get reported with full context, then the loop stops.
