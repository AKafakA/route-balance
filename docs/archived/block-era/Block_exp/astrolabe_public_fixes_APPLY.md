# Apply the AKafakA/Astrolabe README-and-plotter fixes

This patch fixes the public-repo issues that broke README reproduction:
- 5 plotter `DATA_DIR` defaults pointed at non-existent `experiment_output/data/extra_ablation/...`; now match the paths the experiment scripts actually write.
- `po2_plot.py` was missing `--data-dir`; now accepts it.
- `po2_plot.py --ttft-slo` default was `3.0`; corrected to the paper's `10.0`.
- `llumnix_comparison_plot.py` expected subdirs named `block/` and `llumnix/` with `qps{N}/` children; the script writes `block_comparison/` and `llumnix_comparison/` with `block_qps{N}/` / `llumnix_qps{N}/` children — plotter loader now handles the actual layout.
- README §3.3 + §9 commands updated to point at the corrected paths.

**No anonymity-sensitive strings introduced** (no author names, affiliations, emails, hostnames, or internal usernames). Verified by grep against the diff.

## To apply + push (run on your machine where your AKafakA git identity is configured)

```bash
# 1. Clone (if you don't already have it)
git clone git@github.com:AKafakA/Astrolabe.git
cd Astrolabe

# 2. Apply the patch
git apply --check /path/to/astrolabe_public_fixes.patch   # dry-run, should print nothing on success
git apply         /path/to/astrolabe_public_fixes.patch

# 3. Verify the plotters still parse
python -m py_compile experiments_analysis/{po2,burstiness,cpu_overhead,prediction_error,llumnix_comparison}_plot.py
python experiments_analysis/po2_plot.py --help | grep -E 'data-dir|ttft-slo'

# 4. Commit + push (use whatever identity you want visible in repo history)
git add -A
git commit -m "Fix plotter DATA_DIR paths + README to match experiment-script output

- 5 plotter defaults now match what the experiment scripts actually write
  (drop the non-existent experiment_output/data/extra_ablation/ middle path).
- po2_plot.py: add missing --data-dir; correct --ttft-slo default 3.0 -> 10.0
  (matches the paper's fixed SLO).
- llumnix_comparison_plot.py: load from block_comparison/ + llumnix_comparison/
  subdirs (with block_qps{N}/llumnix_qps{N}/ children) as full_comparison.sh
  actually writes.
- README sections 3.3 and 9 updated to use the corrected paths."
git push origin main
```

## What stays your decision

- **Push under which identity** — your AKafakA identity (recommended) or any other you control. I deliberately did NOT pre-set git author/committer to avoid impersonation.
- **Whether to push now during double-blind review** — see the anonymity note in chat. Pushing now is fine ONLY if (a) the repo being publicly viewable under "AKafakA" is acceptable to your target venue's policy, or (b) review is over / camera-ready.

## Source of truth

The patch was generated from a clean clone of `github.com/AKafakA/Astrolabe` at commit `2e5fad6` (HEAD of `main` as of audit). If upstream has new commits, run `git apply` and resolve any small conflicts; the changes are localized to 6 files.
