# cara_paper/ — Master Index

**Maintenance rule**: when adding a new doc, classify it (active / handoff / archive / cluster-specific) and place it in the right subdir. When replacing an active doc, move the old version to `archive/`. When ending a working session with material changes, write a new handoff in `handoffs/YYYY-MM-DD_handoff.md` and update the "latest handoff" pointer in `MEMORY.md` and below.

---

## 🟢 ACTIVE (currently maintained, source of truth)

### Methodology source of truth — `codex/`
- **`codex/apr15_baseline_comparison_design.md`** — primary methodology design (revision 3)
- **`codex/apr15_baseline_comparison_claude_handoff.md`** — short brief (revision 3)
- **`codex/CODEX_REVIEW_PACKAGE.md`** — codex review entry point + 17 locked decisions + change-log
- **`codex/cara-data-and-experiments.md`** — data + experiment layout reference (kept actively in sync)
- **`codex/experiment_playbook.md`** — bench command playbook

### Top-level + `active/`
- **`ROADMAP_TO_SUBMISSION.md`** — top-level submission roadmap
- **`active/CARA_PROJECT_STATUS.md`** — overall project status
- **`active/PAPER_UPDATES_TODO.md`** — paper section TODO list (§-level)
- **`active/SUBMISSION_PLAN.md`** — strategic submission plan
- **`active/EXPERIMENT_PLAN_CLOUDLAB.md`** — CloudLab Phase 2 (Apr 26 - May 6) plan
- **`active/APR26_PREFLIGHT.md`** — Apr 26 preflight checklist
- **`claude/README.md`** — directory pointer (small)
- **`claude/tools.md`** — env vars, HF tokens, SSH targets (active reference)
- **`plans/related_work_survey.md`** — related-work paper survey (used for §6)

### Cluster-specific live state — `smoke_test_apr_13/`
- **`smoke_test_apr_13/PLAN.md`** — Apr 13-17 cluster plan
- **`smoke_test_apr_13/TRACKER.md`** — sweep run tracker
- **`smoke_test_apr_13/BASELINE_IMPL_TRACKER.md`** — implementation + cycle log (single live queue)
- **`smoke_test_apr_13/scripts/`** — sweep + aggregation scripts (active code)
- **`smoke_test_apr_13/results/`** — sweep outputs (clean-bench results from Apr 26 sweep will land here)
- **`smoke_test_apr_13/configs/`** — model deployment + scheduler configs

### Paper draft (do NOT touch unless explicit)
- **`cara_new.tex`** — NeurIPS draft
- **`new_greedy_cara.tex`** — alternate framing draft
- **`sample.bib`** — bibliography
- **`acm.bst`, `usenix.sty`, `usenix.tex`** — formatting

---

## 📜 HANDOFFS — chronological session bridges (`handoffs/`)

Format: `handoffs/YYYY-MM-DD_handoff.md`. Always read the latest one first when picking up a session.

| 🟢🟢 2026-06-11 | `handoffs/2026-06-11_v2_campaign_close.md` | **READ FIRST — v2 campaign CLOSE-OUT.** 5-day synthesis (Jun 6–11): all experiments + conclusions, paper converged 7/7/7 (commit `4cfe066`), arm-4 reframe, rb rm-rf incident + recovery + provenance audit, lessons, next-slot list. Cluster RELEASED. Rolling operational log = `2026-06-07_v2_rerun_handoff.md`. |

| Date | File | Headline |
|---|---|---|
| 🟢 2026-05-05 | `handoffs/2026-04-28_handoff.md` | **READ FIRST.** Bottom entry "2026-05-05 11:05 UTC — overnight grid COMPLETE + aggregated" is authoritative. 105/105 cells, 0 errors, 371,070 reqs synced + plotted. Active 13-instance cluster `asdwb-302610` (NO P100; c240g5 entries in older `host_configs.json` are zombies — `model_deployment.json` is source of truth). Scheduler=`d8545-10s10301`, vLLM head port=**8000**, cara_serve port=**8200**. Aggregator outputs in `smoke_test_apr_13/results/main_table_full_3534/cara_final_e2e/_aggregate/`. |
| 2026-04-19 | `handoffs/2026-04-19_handoff_vast_sweep.md` | Vast sweep state |
| 2026-04-15 (v2 PM) | `handoffs/2026-04-15_handoff_v2.md` | Phase 1 preflight COMPLETE (Apr 16 03:30 UTC). 3 runs, 10 bugs fixed, 3 paper-facing findings. (Older — defer to 2026-04-28 for cluster topology.) |
| 2026-04-15 (AM) | `handoffs/2026-04-15_handoff.md` | Revision-3 methodology lock. Clean bench. Phase 1 preflight / Phase 2 paper-facing. |
| 2026-04-14 | `handoffs/2026-04-14_handoff.md` | Phase A baseline implementation snapshot (pre-revision-3) |
| 2026-04-13 | `handoffs/2026-04-13_handoff.md` | Pre-Apr-15 design discussion |
| 2026-04-11 | `handoffs/2026-04-11_handoff.md` | Phase 1 (model exploration) CLOSED |
| 2026-04-10 | `handoffs/2026-04-10_handoff.md` | Predictor eval verified |
| 2026-04-07 | `handoffs/2026-04-07_handoff.md` | All training complete |
| 2026-04-05 | `handoffs/2026-04-05_handoff.md` | Training plan + matrix |
| 2026-04-02 | `handoffs/2026-04-02_handoff.md` | DeepEval scoring + P100 smoke |
| 2026-03-31 | `handoffs/2026-03-31_handoff.md` | 5 eval targets + Prometheus |
| 2026-03-30 | `handoffs/2026-03-30_handoff.md` | Mid-training checkpoint |
| 2026-03-29 | `handoffs/2026-03-29_handoff.md` | Filter implementation |
| 2026-03-28 | `handoffs/2026-03-28_handoff.md` | Batch scheduling + normalization |
| 2026-03-27 | `handoffs/2026-03-27_handoff.md` | Code-paper audit |

---

## 🗄 ARCHIVE — frozen historical reference (`archive/`)

These docs are **historical snapshots** — kept for audit but not actively maintained. Newer docs supersede them. Cross-references inside archived docs may point at stale paths; use the active source-of-truth docs for current state.

### `archive/claude/` — predictor + scheduler design history
- `ACTION_PLAN_MARCH27.md` — late-March action plan (superseded)
- `BATCH_SCHEDULING_IMPLEMENTATION.md` — March 27 batch scheduler implementation log
- `CHANGELOG_LLM_JUDGE.md` — judge scoring history
- `CODE_PAPER_AUDIT_MARCH27.md` — original code-paper audit (62 items, mostly resolved)
- `DEPLOYMENT_WORKFLOW.md` — January deployment doc (superseded by setup.sh)
- `A100_DEPLOYMENT_GUIDE.md` — January A100 deployment guide
- `EXPERIMENT_STATUS_MARCH2026.md` — March 20 status snapshot
- `EXPERIMENT_QUICK_REFERENCE.md` — January quick ref
- `IMPLEMENTATION_FIX_PLAN.md` — early implementation fix log
- `IMPLEMENTATION_TASKS_V2_DESIGN.md` / `IMPLEMENTATION_TASKS_DETAILED.md` — task-decomposition designs
- `INSTANCE_STATS_API.md` — instance_stats endpoint design (now landed in code)
- `LATENCY_PREDICTOR_DESIGN.md` — Phase 1 predictor design
- `LLM_JUDGE_IMPROVEMENTS.md` — judge improvement notes
- `LTR_PROJECT_INIT.md` — Learned-to-Rank future-work proposal
- `MODEL_COMPARISON_MARCH29.md` — March 29 model comparison
- `OVERNIGHT_MONITOR.md` — overnight log
- `P100_DEPLOYMENT_GUIDE.md` — P100 deployment guide
- `PREDICTOR_IMPLEMENTATION_REVIEW.md` / `PREDICTOR_TRAINING_PLAN.md` — predictor build history
- `PROPOSAL_LEARNED_GLOBAL_SCHEDULING.md` — future-work LTR proposal
- `TRAINING_PLAN_FINAL.md` / `TRAINING_PLAN_MARCH29.md` — training plans (executed, archived)
- `TASK_1_4_DESIGN.md` — Task 1.4 design (executed)
- `learned_latency_predictor.md` — predictor design notes
- `review_comments_for_implementation_predictor_v1.md` — review comments

### `archive/claude/apr_14/` — Apr 14 working session artifacts
- `APR14_EXTENSIONS.md`, `MEANWHILE_MENU.md`, `OVERHEAD_OPTIMIZATIONS.md`, `PHASE_EF_PLAN.md`, `REVIEW.md`

### `archive/codex/` — codex design iterations (pre-revision-3)
- `cara-objective-design.md` — March objective design
- `codex-cara-plan.md` — January codex-cara plan

### `archive/plans/` — historical reports + log files
- `code_review_march26.md`, `experiment_report_march19.md`, `experiment_report_march23_25.md`
- `failed_experiments.md`, `gpu_vm_recovery_march25.md`
- `interactive_debugging_march26.md`, `paper_review_march26.md`
- `paper_vs_code_gaps.md`, `training_plan_march23.md`
- `*.log` files (study logs, gpu monitor logs)

---

## Suggested workflow conventions

- **New session pickup**: read `MEMORY.md` → see latest `handoffs/YYYY-MM-DD_handoff.md` → follow its read-order list.
- **Material session change**: at end of session, write `handoffs/YYYY-MM-DD_handoff.md`, update `MEMORY.md`'s "Active Plans — READ THIS FIRST" pointer, append cycle row to `smoke_test_apr_13/BASELINE_IMPL_TRACKER.md`.
- **Active doc edit**: edit in place; the file path doesn't change. If the doc is becoming obsolete, move to `archive/` and add a "superseded by X" header at top.
- **Archive a doc**: prepend `> **ARCHIVED — see <pointer>** This doc is a historical snapshot.` then `mv` it under `archive/`.
- **Code references**: prefer paths relative to the repo root; sed-replaceable across the doc tree.
