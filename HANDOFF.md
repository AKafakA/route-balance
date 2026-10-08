# RouteBalance: handoff, 8 October 2026

For the next session. Read it with the project rules (`.claude/CLAUDE.md`), the global rules (`~/.claude/CLAUDE.md`,
`~/.claude/shared-experimental-rules/`) and the memory index. Machines are named as in the global rules: the
operator (the VPS that holds the repositories), the GPU VM, the HPC cluster.

## Where the project stands

- **MLSys 2027, due 30 October 2026.** The paper is at version 1.1 rc10 (8 Oct, night). The final runs (the lease of 28 Sept
  to 5 Oct: 1,457 runs, all valid) are archived. The code is released in three branches, and the reviewer packs are
  built. Every backup was checked by sha256 on 8 Oct. Nothing is running on any machine.
- **Before submitting: the paper's code link.** The abstract links `https://anonymous.4open.science/r/route-balance-3ECA`.
  It still serves the snapshot made for the NeurIPS submission (top level `route_balance/`, `AGENTS.md`, `REPRODUCE.md`,
  `data/`, `reproduce/`; read from the mirror's public file list on 8 Oct), not the submission code. It follows no
  current branch of the GitHub repository. The owner points it at the submission code in the anonymizer's dashboard
  (the `submission` branch, commit 984eb1a), or hosts the submission archive another anonymous way.

## The paper

- Repository `~/Code/paper/routebalance` (git, no remote; bundled on Drive, below). Version 1.1 rc10 (8 Oct): commit
  ee71e41, tag `v1.1-rc10`, snapshot `versions/1.1.0-rc10/` (reader PDFs, the Overleaf ZIP, every check's output, the
  round's reviews, `SHA256SUMS`). The body ends on page 10. rc6 over rc5: no paragraph ends in a single-word line, ALPS
  is cited among the request schedulers, the Testbed sentence gives the V100's reason (`reviews/2026-10-08_runts-alps/`).
  rc7 over rc6: the round's range measured on the archived decision logs (`tools/round_range.py`,
  `data/round_range.json`, both reviews) and explained in two sentences of §5.4 with a table in Appendix I; four
  duplicate sentences removed, so page 10 is unchanged; the day's audit, MLSys review, triage and rebuttal notes in
  `reviews/2026-10-08_mlsys-review-audit/`. rc8 over rc7: the review round on rc7 by the three plain readers (MLSys 4 of 6,
  readability 7, story 8, AI style 3, visual 0, desk rejection 0): Table A8's lead column defined, a restating clause and
  a pronoun fixed; the readers' adopt candidates wait for the owner in that folder's `ROUND_LOG.md`. rc9 over rc8
  (owner 8 Oct): BOute (MLSys 2026) named in §2's Placement paragraph as the offline deployment search that brings in
  model routing (a fixed workload trace, a cluster simulator, P95 latency against quality under a price budget), its
  reference added, no Table 1 row; for the fit Figure 4 (Gemma) is drawn at 2.20 in, no `\looseness`, the body still
  ends on page 10 (`ROUND_LOG.md`, ledger rows L123 and L124). rc10 over rc9 (owner 8 Oct): Figure 4's caption in one line (his text), the keys of
  Figures 2 and 4 in two lines from the left edge with the hollow entry on one line, Figure 4 at 2.40 in (2.50 in
  overflows; ledger rows L125 to L128). The paper data pack is rebuilt as v1.3 for rc10 (below). The paper
  repository's Drive bundle predates rc7.
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
| the reviewer packs, version 1.3 (8 Oct night, for the paper at rc10) | the GPU VM `~/rb-analysis-nfs/packs/release-v1.3/` with `SHA256SUMS`; Drive `release-v1.3/` | code `f1b969d7…` (unchanged, commit 984eb1a), paper data `f46955b8…` (the two plot tools at rc10: Figure 4 at 2.40 in with a two-line left key, Figure 2's key in two lines; the expected figures rebuilt), meta-analysis `a3bc3c4d…` (unchanged v1.2); built twice to the same bytes, verified from a fresh unpack on the GPU VM with an environment from the pack's requirements (manifest 77 files, PASS: 24 reference outputs), identity scan 0 hits; Drive sha256 = local, 4 of 4 files; the Drive `BACKUP_INDEX.md` and `.tsv` updated (the `.tsv` also gains the v1.2 rows it lacked). The plot tools live only in the paper repository and this pack: the three code branches hold no figure code and are unchanged. Release folder commit 18d23e5 |
| the reviewer packs, version 1.2 (8 Oct evening, for the paper at rc8) | the GPU VM `~/rb-analysis-nfs/packs/release-v1.2/` with `SHA256SUMS`; Drive `release-v1.2/` | code `f1b969d7…` (unchanged, commit 984eb1a), paper data `bc9bf151…` (gains `round_range.json`, the generator at rc8, Table A8), meta-analysis `a3bc3c4d…` (gains the round-range tool and the 18 decision logs it reads); each built twice to the same bytes, verified from a fresh unpack on the GPU VM (24 reference outputs, 19 files), masked identity scan as v1.1; Drive sha256 = local, 3 of 3 (8 Oct) |
| the reviewer packs, version 1.1 (6 Oct, superseded) | the GPU VM `~/rb-analysis-nfs/packs/release-v1.1/` with `SHA256SUMS`; Drive `release-v1.1/` | code `f1b969d7…`, paper data `81ca72e8…`, meta-analysis `983dc622…`; Drive = the GPU VM, 4 of 4 (8 Oct) |
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

- **The operator:** 61 GB free of 154 GB after the 8 Oct cleanup (owner's order). Removed, each after a verified
  backup: `~/Code/llm/Block` and `~/Code/llm/Block_exp` (the git bundle of Block and the full trees as packs on Drive,
  `routebalance-backup/archive-block-era/`; 533 of their documents migrated to `docs/archived/block-era/` here),
  `~/Code/llm/rb-exp` (the NeurIPS-era results: packs on Drive there and on CSD3 `hpc-work/llm/rb-exp-nips-packs/`),
  the lease-time run copies of `local/` (13 GB: packs `lease-local-pack-01` to `06` on CSD3
  `route-balance-lease-20260928/operator-local-packs/` and on Drive `routebalance-backup/lease-operator-local/`,
  both verified by sha256; the pack records stay in `~/data/backup-staging-rb/packs/`), the pip, uv and npm caches, the replay venv and
  the SWE-bench Docker images. `local/` keeps its scripts, logs, manifests, renders and calibration tables (161 MB), plus the eight pinned inputs of
  the paper's dataset table restored from the CSD3 packs on 8 Oct night (`local/lease/data/` six prompt and corpus
  files, `local/lease/bundle/inputs/` the two training-row files, 188 MB, every pin checked): the paper's
  `tools/snapshot.py` regenerates `generated/datasets.tex` from them and stops without them.
  Never copy run folders here; size before copying.
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
3. Drive: the old folder `route-balance-lease-backup-2026-10` at the Drive root holds nothing of the backup's any
   more (all 82 files moved and checked); it keeps something rclone did not make, so the owner deletes it in the
   browser after a look. rclone's shared client id is being retired during 2026 (make an own client id when uploads
   start failing). vSkipper's backups are one folder too since 8 Oct (`project notes/vskipper`, packs 01-21; the
   shared rule `backup_to_google_drive.md` lists both projects' folders).
4. When no longer needed: the GPU VM's proof folders (`~/rb-ref`, `~/rb-gpu`, about 50 GB).
