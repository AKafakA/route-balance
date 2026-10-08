# RouteBalance: handoff, 8 October 2026

For the next session. Read it with the project rules (`.claude/CLAUDE.md`), the global rules (`~/.claude/CLAUDE.md`,
`~/.claude/shared-experimental-rules/`) and the memory index. Machines are named as in the global rules: the
operator (the VPS that holds the repositories), the GPU VM, the HPC cluster.

## Where the project stands

- **MLSys 2027, due 30 October 2026.** The paper is at version 1.1 rc5 (6 Oct). The final runs (the lease of 28 Sept
  to 5 Oct: 1,457 runs, all valid) are archived. The code is released in three branches, and the reviewer packs are
  built. Every backup was checked by sha256 on 8 Oct. Nothing is running on any machine.
- **Before submitting: the paper's code link.** The abstract links `https://anonymous.4open.science/r/route-balance-3ECA`.
  It still serves the snapshot made for the NeurIPS submission (top level `route_balance/`, `AGENTS.md`, `REPRODUCE.md`,
  `data/`, `reproduce/`; read from the mirror's public file list on 8 Oct), not the submission code. It follows no
  current branch of the GitHub repository. The owner points it at the submission code in the anonymizer's dashboard
  (the `submission` branch, commit 984eb1a), or hosts the submission archive another anonymous way.

## The paper

- Repository `~/Code/paper/routebalance` (git, no remote; bundled on Drive, below). Version 1.1 rc5: commit 5033f6b,
  tag `v1.1-rc5`, snapshot `versions/1.1.0-rc5/` (reader PDFs, the Overleaf ZIP, every check's output, the round's
  reviews, `SHA256SUMS`). The body ends on page 10.
- Its `HANDOFF.md` was last updated on 5 Oct (rc2). The rounds since (rc3 to rc5: the rc2 review's fixes, the
  related-work audit, the flow rewrite, the page fit, the merged figures, Table 5 as a figure, the two-column
  appendix) are in its commit log and in `reviews/2026-10-05_related-audit/` and `reviews/2026-10-06_flow-rewrite/`.
- Every update: edit main at the root, log the edits old to new, snapshot `versions/<x.y.z-rcN>/` with that round's
  reviews, recompute its `SHA256SUMS`, tag `v<x.y>-rcN`. References are never edited.
- The paper's inputs are rebuilt from the archived runs (the GPU VM's `~/rb-analysis-nfs/final36/reports/`).

## The code: `route-balance`, three branches (GitHub remote `origin`, private)

| branch | commit | worktree | what it is |
|---|---|---|---|
| `dev` | this commit | `~/Code/llm/route-balance` | the development record: all QA (tests, gates, `make check`), internal docs and configs, the lease's operator records in `local/` (git-ignored). Only stale, never-run code and configs may be cleaned. |
| `submission` | 984eb1a | `~/Code/llm/route-balance-submission` | the code that ran (the service and the run scripts byte-identical to dev's), the final render `configs/render36/`, the reproduction guides `docs/reproduce/`, the README. Published only as `git archive --format=tar.gz --prefix=routebalance/ 984eb1a` (= `routebalance-code-v1.1.tar.gz`, sha256 `f1b969d7…`). |
| `ref` | ce37508 | `~/Code/llm/route-balance-ref` | the code rewritten for reading and proven to do what the code that ran did (below). |

- What `ref` changed: the record readers lost their fallbacks for records older than the final runs; three options
  no run used are gone (BEST-Route's older four-way classifier, the dispatchers' in-flight count, the older
  pending-decode mode); one runner (`python -m runner rows|cell|parity|status|retire`) replaces the run driver and
  four run scripts, with the same steps, order, refusals and exit codes; the scheduler's server module is split into
  six modules, the bodies moved unchanged; comments and messages carry no development history; `CODE_TOUR.md` walks
  one request end to end; dev's tests are ported; `proofs/` holds the proofs. One behaviour change, in a failure
  path: a row list that names no row (`--rows ,`) exits 2 (usage), where the old script exited 3.
- Size (lines of `.py` and `.sh`): the code a reviewer reads is 15,618 in `submission` and 15,819 in `ref`; `ref`
  adds 12,195 lines of tests, 430 of repository checks and 1,247 of proofs. The code tour's one-hour path is 1,418
  lines.
- Also on GitHub: the review, rebuild, backup and debug branches of late September; the tag `lease-2026-09-28` (the
  freeze of the lease's scope); the tag `archive/main-2026-06-18` (the June arXiv-era history, 12 commits that no
  branch holds any more).

## The ref branch's proofs (all passed, 7-8 Oct, on the GPU VM)

| proof | what was compared | result |
|---|---|---|
| records and verdicts | all 1,457 archived runs re-read by the ref code, their run gates and verdicts recomputed | 1,457 equal to the archived verdicts; the output file byte-identical to the code that ran's |
| reports | the four reports rebuilt by the ref analysis, after its parity gate ran over the 8 comparison groups | every file (145, 145, 23, 23) and log byte-identical to the run bundle's |
| routers | 644 runs, 2,162,792 requests: model, score and reason | 0 runs differ; the chosen model equals the logged one for every request |
| estimator, one request | 246 runs, 1,168,551 requests | 0 differ; every logged decode and feature token count reproduced |
| dispatchers | 779 runs, 2,967,245 choices | 0 differ |
| estimator, each round | 678 runs, 1,210,741 rounds, and the 678 saved estimate files | 0 differ; the files identical |
| rounds | 678 runs, 1,210,741 rounds, 2,310,834 decisions | 0 differ |
| GPU end to end | the RTX 8000, vLLM 0.18.1, Qwen2.5-3B and 7B, 11 configurations, each tree's own run command | 11 of 11 valid under both codes; parity passed; the timing-free choices 40 of 40 |
| vLLM Semantic Router | its test profile (two Qwen2.5-0.5B engines, the authors' Docker stack), through both drivers | both valid; prompt, arrival, model and instance 40 of 40 |
| tests | unit 932, contract 137, gates 375, vLLM-SR bench 44 | all pass; lint, hygiene and static checks clean |

- The replays compare the two codes on one machine: the code that ran reproduces 687,107 of the 2,310,834 logged
  length estimates exactly on the GPU VM, the rest within 6.8% (the lease ran torch with 48 threads on 96-CPU nodes,
  the replay with 8).
- BEST-Route and AvengersPro are not in the GPU run (they need the four-model fleet); the router replay covers them.
- The harness: `proofs/README.md`; the scripts and outputs: the GPU VM's `~/rb-ref` and `~/rb-gpu`.

## The data and the backups

| what | where | checked |
|---|---|---|
| the final runs (1,457) | the raw bundle `rb-runs-v1.1.tar.zst` (sha256 `27e989fb…`): the GPU VM `~/rb-analysis-nfs/packs/raw-v1.1/`, the HPC cluster, Drive `raw-bundle-v1.1/`. Unpacked: the GPU VM `~/rb-analysis-nfs/packs/rb-runs-v1.1/` and `~/rb-analysis-nfs/final36/` | Drive = the GPU VM, 7 of 7 files (8 Oct) |
| the reviewer packs | the GPU VM `~/rb-analysis-nfs/packs/release-v1.1/` with `SHA256SUMS`; Drive `release-v1.1/` | code `f1b969d7…`, paper data `81ca72e8…`, meta-analysis `983dc622…`; Drive = the GPU VM, 4 of 4 (8 Oct) |
| the lease's record | the HPC cluster's work folder `route-balance-lease-20260928/` (the incremental archives, the bundles, the cell trees, the logs); Drive, every folder as one tar (30, listed in `BACKUP_INDEX.md`); the GPU VM's `~/rb-lease-fallback-20261005/` (9 of the tars) | per-file sha256 in `local/lease/backup-records/manifests/`; read `ARCHIVE_NOTE_2026-10-05` before using the 70 reruns' files |
| the code and the records | GitHub (`route-balance`); Drive `code-pack-02-git-bundles/` (route-balance, route-balance-paper, the paper repository, the private release tools, the earlier campaigns' repository, as git bundles) | Drive = local, 9 of 9 (8 Oct) |

- **Drive:** one folder for the project, `project notes / routebalance / routebalance-backup`. Its `BACKUP_INDEX.md`
  lists each item and what was removed on 8 Oct (stale, with a newer or full copy elsewhere; in the Drive trash
  until about 7 Nov). From the operator:
  `rclone lsf "gdrive,root_folder_id=1yUUX6xcRUpaDEmdGjs69n-UdKFUh5HRP:routebalance-backup"`. The remote's scope is
  `drive.file` (it sees only what rclone made); Drive keeps a server-side sha256 (`rclone hashsum sha256`).
- All 18 artifact folders the final configurations load are in the backup (the cluster's tars on Drive: `bundle3`, `bundle4g`,
  `bundle-d208`, `archives-20261002`, `xgb`); the GPU VM's `~/rb-ref/artifacts` holds a working copy.

## Machines

- **The operator:** 4.3 GB free of 154 GB, under the 5 GB floor. Never copy run folders here; size before copying.
- **The GPU VM:** idle (no engine, container or proof). `~/rb-analysis-nfs` 22 GB (the final runs and the packs: keep);
  `~/rb-ref` 21 GB and `~/rb-gpu` 28 GB (the ref proofs: tree copies, outputs, the scheduler and engine environments,
  the Hub caches); the vLLM-SR test setup in the local scratch (`rb-vsr`, 9.5 GB) and `/var/tmp` (8.2 GB, the
  classifier weights). The proof folders can go when they are no longer wanted.
- **The HPC cluster:** the lease's record (above). Compute only through SLURM (INTR for debugging).
- **CloudLab:** the lease's 13 nodes were released on 5 Oct, 08:42 BST; nothing is reserved.

## Records

- Decisions: `~/Code/llm/route-balance-paper/mlsys_2027/DECISION_LOG.md`.
- The lease: `~/Code/llm/route-balance-paper/mlsys_2027/lease/LEASE_HANDOFF.md`; the operator's lease folder
  `local/lease/` here (git-ignored).
- The release: `~/Code/llm/route-balance-release-private/RELEASE_LOG.md`, with the private tools (`mask.py`,
  `scan_tree.py`). The private term lists are git-ignored there and never shipped; every dump goes through `mask.py`.
- The ref branch's plan: `~/.claude/plans/enter-into-the-plan-quiet-flask.md`.
- The paper's review rounds: `~/Code/paper/routebalance/reviews/`.

## Open items

1. The paper's code link (above), before submitting.
2. The paper's final read and the submission, by 30 October.
3. Drive: the old folder `route-balance-lease-backup-2026-10` at the Drive root shows empty to rclone but holds
   something rclone did not make; check it in the browser. rclone's shared client id is being retired during 2026
   (make an own client id when uploads start failing). vSkipper's backup folder is still at the Drive root.
4. When no longer needed: the GPU VM's proof folders (`~/rb-ref`, `~/rb-gpu`, about 50 GB).
